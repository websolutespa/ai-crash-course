function product_list() {
  // Gestione prodotti
  const products = [
    { id: 1, name: 'Laptop', price: 999.99 },
    { id: 2, name: 'Smartphone', price: 499.99 },
    { id: 3, name: 'Headphones', price: 79.99 }
  ];

  products.forEach(product => {
    console.log(`Prodotto: ${product.name}, Prezzo: €${product.price}`);
  });
}
