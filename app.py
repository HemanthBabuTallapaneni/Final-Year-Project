from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import algorithms
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a secure secret key
app.config['MONGO_URI'] = 'mongodb://localhost:27017/e_commerce'  # MongoDB connection URI
mongo = PyMongo(app)

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    products = list(mongo.db.products.find())  # Convert cursor to list
    user_id = session.get('user_id')

    if 'user_id' in session:
        # Fetch liked products for recommendations
        liked_products_aggregate = mongo.db.likes.aggregate([
            {'$match': {'user_id': user_id}},  # Match the current user's likes
            {'$lookup': {
                'from': 'products',  # Join with the products collection
                'localField': 'product_ids',  # Use the array of product IDs
                'foreignField': '_id',  # Match with the _id field in products
                'as': 'products'  # Store the joined products in a field called 'products'
            }},
            {'$unwind': '$products'}  # Unwind the array of products
        ])
        viewed_products_aggregate = mongo.db.views.aggregate([
            {'$match': {'user_id': user_id}},  # Match the current user's likes
            {'$lookup': {
                'from': 'products',  # Join with the products collection
                'localField': 'product_ids',  # Use the array of product IDs
                'foreignField': '_id',  # Match with the _id field in products
                'as': 'products'  # Store the joined products in a field called 'products'
            }},
            {'$unwind': '$products'}  # Unwind the array of products
        ])
        cart_count = 0
        if user_id:
            cart_count = mongo.db.cart.count_documents({'user_id': user_id})
        
        # Extract the products from the aggregation result
        liked_products = [item['products'] for item in liked_products_aggregate]
        viewed_products = [item['products'] for item in viewed_products_aggregate]
        if liked_products and viewed_products:
            products = algorithms.get_recommendations(products, user_id, liked_products,recent_views=viewed_products)
            if user_id:
                return render_template('index.html', products=products,cart_count=cart_count)
        return render_template('index.html', products=products)
    return render_template('index.html', products=products)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    user_id = session.get('user_id')

    products = list(mongo.db.products.find({'name': {'$regex': query, '$options': 'i'}}))  # Convert cursor to list
    cart_count = 0
    if user_id:
        cart_count = mongo.db.cart.count_documents({'user_id': user_id})
    return render_template('search.html', products=products, query=query,cart_count=cart_count)


# Configure upload folder for images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed file extensions for images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/add_product', methods=['GET', 'POST'])
@login_required
def add_product():
    if request.method == 'POST':
        try:
            # Ensure the user is logged in
            if 'user_id' not in session:
                flash('You must be logged in to add a product', 'error')
                return redirect(url_for('login'))

            # Get form data
            name = request.form.get('name')
            description = request.form.get('description')
            price = float(request.form.get('price'))
            category = request.form.get('category')
            brand = request.form.get('brand')
            color = request.form.get('color')
            size = request.form.get('size')
            inventory = int(request.form.get('inventory'))
            image = request.files.get('image')

            # Validate required fields
            if not name or not price or not image or not category or not color or not size:
                flash('Name, price, category, color, size, and image are required', 'error')
                return redirect(url_for('add_product'))

            # Validate image file
            if not allowed_file(image.filename):
                flash('Invalid file type. Allowed types: png, jpg, jpeg, gif', 'error')
                return redirect(url_for('add_product'))

            # Insert product into MongoDB
            product_data = {
                'name': name,
                'description': description,
                'price': price,
                'category': category,
                'brand': brand,
                'color': color,
                'size': size,
                'inventory': inventory,
                'likes': 0,
                'views': 0,
                'is_new': True,  # Set is_new to True for newly added products
                'added_by': session['user_id'],  # Store the user ID who added the product
                'created_at': datetime.now()
            }
            product_id = mongo.db.products.insert_one(product_data).inserted_id

            # Save the image with the product ID as the filename
            if image:
                file_extension = image.filename.rsplit('.', 1)[1].lower()  # Get file extension
                filename = f"{product_id}.{file_extension}"  # Use product ID as filename
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)

                # Update the product with the image path
                mongo.db.products.update_one(
                    {'_id': product_id},
                    {'$set': {'image_url': f"static/uploads/{filename}"}}  # Include "static/" in the path
                )

            flash('Product added successfully!', 'success')
            return redirect(url_for('index'))

        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(url_for('add_product'))

    # Render the add product form for GET requests
    return render_template('add_product.html')


@app.route('/view/<product_id>', methods=['POST'])
@login_required
def view_product(product_id):
    try:
        user_id = session.get('user_id')
        product_id = ObjectId(product_id)

        # Ensure user views tracking exists
        mongo.db.views.update_one(
            {'user_id': user_id},
            {'$addToSet': {'product_ids': product_id}},  # Prevent duplicate views
            upsert=True
        )

        # Increment the view count for the product
        mongo.db.products.update_one(
            {'_id': product_id},
            {'$inc': {'views': 1}}
        )

        # Get updated views count
        product = mongo.db.products.find_one({'_id': product_id})
        views_count = product.get('views', 0)

        return jsonify({'status': 'success', 'views': views_count})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/like/<product_id>', methods=['POST'])
@login_required
def like_product(product_id):
    try:
        user_id = session.get('user_id')
        product_id = ObjectId(product_id)

        # Ensure user likes tracking exists
        mongo.db.likes.update_one(
            {'user_id': user_id},
            {'$setOnInsert': {'product_ids': []}},
            upsert=True
        )

        # Check if the product is already liked
        like = mongo.db.likes.find_one({
            'user_id': user_id,
            'product_ids': product_id
        })

        if like:
            mongo.db.likes.update_one(
                {'user_id': user_id},
                {'$pull': {'product_ids': product_id}}
            )
            mongo.db.products.update_one(
                {'_id': product_id},
                {'$inc': {'likes': -1}}
            )
            action = 'unliked'
        else:
            mongo.db.likes.update_one(
                {'user_id': user_id},
                {'$push': {'product_ids': product_id}}
            )
            mongo.db.products.update_one(
                {'_id': product_id},
                {'$inc': {'likes': 1}}
            )
            action = 'liked'

        # Get updated likes count
        product = mongo.db.products.find_one({'_id': product_id})
        likes_count = product.get('likes', 0)

        return jsonify({'status': 'success', 'action': action, 'likes': likes_count})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/cart/add/<product_id>', methods=['POST'])
@login_required
def add_to_cart(product_id):
    try:
        user_id = session.get('user_id')
        product_id = ObjectId(product_id)  # Convert product_id to ObjectId

        # Check if the product exists
        product = mongo.db.products.find_one({'_id': product_id})
        if not product:
            flash('Product not found!')
            return redirect(url_for('index'))

        cart_item = mongo.db.cart.find_one({
            'user_id': user_id,
            'product_id': product_id
        })
        
        if cart_item:
            mongo.db.cart.update_one(
                {'_id': cart_item['_id']},
                {'$inc': {'quantity': 1}}
            )
        else:
            mongo.db.cart.insert_one({
                'user_id': user_id,
                'product_id': product_id,
                'quantity': 1
            })
        
        flash('Product added to cart!')
        return redirect(url_for('index'))
    except Exception as e:
        flash('An error occurred while adding the product to the cart.')
        return redirect(url_for('index'))

@app.route('/account')
@login_required
def account():
    try:
        user_id = session.get('user_id')
        print("Current User ID:", user_id)  # Debug: Print the current user ID
        
        # Fetch the current user's details
        user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        
        # Fetch products added by the current user
        my_products = list(mongo.db.products.find({'added_by': user_id}))
        
        # Fetch orders placed on the current user's products
        orders = list(mongo.db.orders.aggregate([
    {'$match': {'product_owner': str(session.get('user_id'))}},  # Match orders where product_owner is the current user
    {'$lookup': {
        'from': 'products',
        'localField': 'product_id',
        'foreignField': '_id',
        'as': 'product'
    }},
    {'$unwind': {
        'path': '$product',
        'preserveNullAndEmptyArrays': True  # Ensure orders without matching products are not discarded
    }},
    {'$lookup': {
        'from': 'users',
        'localField': 'user_id',
        'foreignField': '_id',
        'as': 'buyer'
    }},
    {'$unwind': {
        'path': '$buyer',
        'preserveNullAndEmptyArrays': True  # Ensure orders without matching buyers are not discarded
    }},
    {'$project': {
        'buyer_name': 1,
        'buyer_address': 1,
        'product_name': '$product.name',
        'product_image_url': '$product.image_url',
        'quantity': 1,
        'status': 1,
        'created_at': 1
    }}
]))
        
        #print("Orders:", orders)  # Debug: Print the fetched orders
        cart_count=0
        if user_id:
            cart_count = mongo.db.cart.count_documents({'user_id': user_id})
        
        return render_template('account.html', user=user, my_products=my_products, orders=orders,cart_count=cart_count)
    
    except Exception as e:
        print("Error:", e)  # Debug: Print the error
        flash('An error occurred while fetching your account details.', 'error')
        return redirect(url_for('index'))

@app.route('/remove_product/<product_id>', methods=['POST'])
@login_required
def remove_product(product_id):
    try:
        user_id = session.get('user_id')
        product_id = ObjectId(product_id)
        
        # Ensure the product belongs to the user
        product = mongo.db.products.find_one({'_id': product_id, 'added_by': user_id})
        if not product:
            flash('Product not found or you do not have permission to remove it.', 'error')
            return redirect(url_for('account'))
        
        # Remove the product
        mongo.db.products.delete_one({'_id': product_id})
        flash('Product removed successfully!', 'success')
        return redirect(url_for('account'))
    except Exception as e:
        flash('An error occurred while removing the product.', 'error')
        return redirect(url_for('account'))
    
@app.route('/checkout', methods=['GET', 'POST'])
@login_required
def checkout():
    if request.method == 'POST':
        user_id = session.get('user_id')
        address = request.form.get('address')
        
        # Fetch the current user's details
        user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if not user:
            flash('User not found.', 'error')
            return redirect(url_for('cart'))
        
        # Fetch cart items
        cart_items = list(mongo.db.cart.find({'user_id': user_id}))
        
        # Create orders for each cart item
        for item in cart_items:
            product = mongo.db.products.find_one({'_id': item['product_id']})
            if product:
                order_data = {
                    'user_id': user_id,  # Buyer's user ID
                    'buyer_name': user['username'],  # Buyer's username
                    'buyer_address': address,  # Buyer's address
                    'product_id': item['product_id'],
                    'product_owner': str(product['added_by']),  # Seller's user ID (as a string)
                    'quantity': item['quantity'],
                    'status': 'Pending',
                    'created_at': datetime.now()
                }
                # Insert the order into the database
                mongo.db.orders.insert_one(order_data)
                print("Order inserted:", order_data)  # Debugging
        
        # Clear the cart
        mongo.db.cart.delete_many({'user_id': user_id})
        
        flash('Order placed successfully!', 'success')
        return redirect(url_for('account'))
    
    return render_template('checkout.html')

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    try:
        user_id = session.get('user_id')
        
        # Update user profile information
        mongo.db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {
                'email': request.form['email'],
                'first_name': request.form['first_name'],
                'last_name': request.form['last_name']
            }}
        )
        
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('account'))
    except Exception as e:
        flash('An error occurred while updating your profile.', 'error')
        return redirect(url_for('account'))

@app.route('/orders')
@login_required
def orders():
    orders = mongo.db.orders.find({'user_id': session.get('user_id')})
    return render_template('orders.html', orders=orders)

@app.route('/wishlist')
@login_required
def wishlist():
    user_id = session.get('user_id')
    cart_count=0
    if user_id:
        cart_count = mongo.db.cart.count_documents({'user_id': user_id})
    # Fetch the user's liked products
    liked_products = mongo.db.likes.aggregate([
        {'$match': {'user_id': user_id}},  # Match the current user's likes
        {'$lookup': {
            'from': 'products',  # Join with the products collection
            'localField': 'product_ids',  # Use the array of product IDs
            'foreignField': '_id',  # Match with the _id field in products
            'as': 'products'  # Store the joined products in a field called 'products'
        }},
        {'$unwind': '$products'}  # Unwind the array of products
    ])
    
    # Extract the products from the aggregation result
    products = [item['products'] for item in liked_products]
    return render_template('wishlist.html', products=products,cart_count=cart_count)

@app.route('/cart')
@login_required
def cart():
    user_id = session.get('user_id')
    cart_count=0
    if user_id:
        cart_count = mongo.db.cart.count_documents({'user_id': user_id})
    cart_items = mongo.db.cart.aggregate([
        {'$match': {'user_id': session.get('user_id')}},
        {'$lookup': {
            'from': 'products',
            'localField': 'product_id',
            'foreignField': '_id',
            'as': 'product'
        }},
        {'$unwind': '$product'}
    ])
    
    total = 0
    items = []
    for item in cart_items:
        item['total'] = item['product']['price'] * item['quantity']
        total += item['total']
        items.append(item)
    
    return render_template('cart.html', cart_items=items, total=total,cart_count=cart_count)

@app.route('/cart/remove/<product_id>', methods=['POST'])
@login_required
def remove_from_cart(product_id):
    try:
        user_id = session.get('user_id')
        product_id = ObjectId(product_id)  # Convert product_id to ObjectId

        # Remove the product from the cart
        mongo.db.cart.delete_one({
            'user_id': user_id,
            'product_id': product_id
        })

        flash('Product removed from cart!')
        return redirect(url_for('cart'))
    except Exception as e:
        flash('An error occurred while removing the product from the cart.')
        return redirect(url_for('cart'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = mongo.db.users.find_one({'username': request.form['username']})
        if user and check_password_hash(user['password'], request.form['password']):
            session['user_id'] = str(user['_id'])
            return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Check if the username already exists
        existing_user = mongo.db.users.find_one({'username': request.form['username']})
        if existing_user:
            flash('Username already exists')
            return redirect(url_for('register'))
        
        # Hash the password
        hashed_password = generate_password_hash(request.form['password'])
        
        # Insert the new user into the database
        user_id = mongo.db.users.insert_one({
            'username': request.form['username'],
            'email': request.form['email'],
            'password': hashed_password,
            'first_name': request.form['first_name'],
            'last_name': request.form['last_name']
        }).inserted_id

        # Initialize likes and views for the new user
        mongo.db.likes.insert_one({
            'user_id': str(user_id),
            'product_ids': []
        })
        mongo.db.views.insert_one({
            'user_id': str(user_id),
            'product_ids': []
        })

        #flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Helper function to check if a product is liked by the current user
def is_product_liked(product_id):
    if 'user_id' not in session:
        return False
    return mongo.db.likes.find_one({
        'user_id': session.get('user_id'),
        'product_ids': ObjectId(product_id)  # Convert product_id to ObjectId
    }) is not None

# Helper function to check if a product is viewed by the current user
def is_product_viewed(product_id):
    if 'user_id' not in session:
        return False
    return mongo.db.views.find_one({
        'user_id': session.get('user_id'),
        'product_id': ObjectId(product_id)  # Convert product_id to ObjectId
    }) is not None

# Make the helper functions available in templates
app.jinja_env.globals.update(is_product_liked=is_product_liked, is_product_viewed=is_product_viewed)

if __name__ == '__main__':
    app.run(debug=True)