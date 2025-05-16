from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from datetime import datetime
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure MongoDB
app.config['MONGO_URI'] = 'mongodb://localhost:27017/e_commerce'  # MongoDB connection URI
mongo = PyMongo(app)

# Configure upload folder for images
UPLOAD_FOLDER = 'static/uploads'  # Save images in the static/uploads folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed file extensions for images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/add_product', methods=['POST'])
def add_product():
    try:
        # Get form data
        name = request.form.get('name')
        description = request.form.get('description')
        price = float(request.form.get('price'))
        category = request.form.get('category')
        brand = request.form.get('brand')
        inventory = int(request.form.get('inventory'))
        image = request.files.get('image')

        # Validate required fields
        if not name or not price or not image:
            return jsonify({'status': 'error', 'message': 'Name, price, and image are required'}), 400

        # Validate image file
        if not allowed_file(image.filename):
            return jsonify({'status': 'error', 'message': 'Invalid file type. Allowed types: png, jpg, jpeg, gif'}), 400

        # Insert product into MongoDB
        product_data = {
            'name': name,
            'description': description,
            'price': price,
            'category': category,
            'brand': brand,
            'inventory': inventory,
            'likes': 0,
            'views': 0,
            'created_at': datetime.now()
        }
        product_id = mongo.db.products.insert_one(product_data).inserted_id

        # Save the image with the product ID as the filename
        if image:
            file_extension = image.filename.rsplit('.', 1)[1].lower()  # Get file extension
            filename = f"{product_id}.{file_extension}"  # Use product ID as filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)

            # Update the product with the image path and file extension
            mongo.db.products.update_one(
                {'_id': product_id},
                {'$set': {'image_url': f"uploads/{filename}", 'file_extension': file_extension}}
            )

        return jsonify({'status': 'success', 'message': 'Product added successfully', 'product_id': str(product_id)}), 201

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)