# Sample Python script
import json

def load_products():
    return [
        {"id": 1, "name": "Laptop", "price": 999.99},
        {"id": 2, "name": "Smartphone", "price": 499.99},
        {"id": 3, "name": "Headphones", "price": 79.99}
    ]

def main():
    products = load_products()
    print("Catalogo prodotti:")
    for p in products:
        print(f"{p['id']}: {p['name']} - €{p['price']}")

if __name__ == "__main__":
    main()
def greet(name):
    print(f"Hello, {name}!")

greet("World")
