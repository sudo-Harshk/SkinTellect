from flask import Flask, render_template_string
import pandas as pd

app = Flask(__name__)

CSV_PATH = "D:/skin-care-complete/dataset/skincare_product_links.csv"

@app.route('/')
def home():
    try:
        df = pd.read_csv(CSV_PATH, encoding='latin1')
        categories = sorted(df['Category'].dropna().unique())

        return render_template_string('''
        <html>
        <head>
            <title>Skin Care Categories</title>
            <style>
                body { font-family: sans-serif; padding: 20px; }
                .category { margin-bottom: 10px; }
            </style>
        </head>
        <body>
            <h1>Browse Products by Category</h1>
            {% for cat in categories %}
                <div class="category">
                    <a href="/products/{{ cat }}">{{ cat }}</a>
                </div>
            {% endfor %}
        </body>
        </html>
        ''', categories=categories)

    except Exception as e:
        return f"<h2>Error loading categories: {str(e)}</h2>"

@app.route('/products/<category>')
def show_products_by_category(category):
    try:
        df = pd.read_csv(CSV_PATH, encoding='latin1')
        filtered = df[df['Category'].str.lower() == category.lower()]
        products = filtered.to_dict(orient='records')

        if not products:
            return f"<h2>No products found in category: {category}</h2>"

        return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>{{ category }} Products</title>
            <style>
                body { font-family: sans-serif; padding: 20px; }
                img { border-radius: 8px; margin-bottom: 5px; }
                .product-card { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 10px; }
                a { text-decoration: none; color: #007BFF; }
            </style>
        </head>
        <body>
            <h1>{{ category }} Products</h1>
            <a href="/">← Back to Categories</a><br><br>
            {% for p in products %}
                <div class="product-card">
                    <img src="{{ p.Image_URL }}" alt="{{ p.Name }}" width="100"><br>
                    <a href="{{ p.Product_URL }}" target="_blank">{{ p.Brand }} - {{ p.Name }}</a>
                </div>
            {% endfor %}
        </body>
        </html>
        ''', products=products, category=category)

    except Exception as e:
        return f"<h2>Error: {str(e)}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
