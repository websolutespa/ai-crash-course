// Sistema di gestione catalogo (catalog) - Demo
// Questo modulo gestisce il catalog dei prodotti

/**
 * Classe principale per la gestione del catalog
 * Il catalog contiene tutti i prodotti disponibili
 */
class Catalog {
  constructor() {
    this.catalog = [];
    this.catalogId = this.generateCatalogId();
    this.catalogName = 'Main Catalog';
    this.catalogVersion = '1.0.0';
  }

  /**
   * Genera un ID univoco per il catalog
   * @returns {string} ID del catalog
   */
  generateCatalogId() {
    const catalogPrefix = 'CATALOG';
    const catalogTimestamp = Date.now();
    const catalogRandom = Math.random().toString(36).substr(2, 9);
    return `${catalogPrefix}-${catalogTimestamp}-${catalogRandom}`;
  }

  /**
   * Aggiunge un prodotto al catalog
   * @param {Object} catalogItem - Item da aggiungere al catalog
   */
  addToCatalog(catalogItem) {
    if (!this.validateCatalogItem(catalogItem)) {
      console.error('Catalog item non valido');
      return false;
    }
    
    // Assegna un catalog item ID
    catalogItem.catalogItemId = this.generateCatalogItemId();
    this.catalog.push(catalogItem);
    console.log(`Aggiunto al catalog: ${catalogItem.name}`);
    return true;
  }

  /**
   * Valida un catalog item prima dell'inserimento
   * @param {Object} catalogItem - Item del catalog da validare
   */
  validateCatalogItem(catalogItem) {
    // Verifica che il catalog item abbia i campi necessari
    const catalogRequiredFields = ['name', 'price', 'category'];
    return catalogRequiredFields.every(field => catalogItem.hasOwnProperty(field));
  }

  /**
   * Genera un ID per un catalog item
   * @returns {string} ID del catalog item
   */
  generateCatalogItemId() {
    const catalogItemPrefix = 'CATALOG-ITEM';
    const catalogItemCounter = this.catalog.length + 1;
    return `${catalogItemPrefix}-${catalogItemCounter}`;
  }

  /**
   * Cerca nel catalog per categoria
   * @param {string} catalogCategory - Categoria da cercare nel catalog
   */
  searchCatalogByCategory(catalogCategory) {
    console.log(`Ricerca nel catalog per categoria: ${catalogCategory}`);
    const catalogResults = this.catalog.filter(
      catalogItem => catalogItem.category === catalogCategory
    );
    return catalogResults;
  }

  /**
   * Rimuove un item dal catalog
   * @param {string} catalogItemId - ID dell'item da rimuovere dal catalog
   */
  removeFromCatalog(catalogItemId) {
    const catalogIndex = this.catalog.findIndex(
      catalogItem => catalogItem.catalogItemId === catalogItemId
    );
    
    if (catalogIndex !== -1) {
      const catalogRemovedItem = this.catalog.splice(catalogIndex, 1);
      console.log(`Rimosso dal catalog: ${catalogRemovedItem[0].name}`);
      return true;
    }
    return false;
  }

  /**
   * Ottiene le statistiche del catalog
   * @returns {Object} Statistiche del catalog
   */
  getCatalogStatistics() {
    const catalogStats = {
      totalCatalogItems: this.catalog.length,
      catalogValue: this.calculateCatalogValue(),
      catalogCategories: this.getCatalogCategories(),
      catalogId: this.catalogId
    };
    return catalogStats;
  }

  /**
   * Calcola il valore totale del catalog
   * @returns {number} Valore totale del catalog
   */
  calculateCatalogValue() {
    return this.catalog.reduce(
      (catalogTotal, catalogItem) => catalogTotal + catalogItem.price,
      0
    );
  }

  /**
   * Ottiene tutte le categorie presenti nel catalog
   * @returns {Array} Array di categorie del catalog
   */
  getCatalogCategories() {
    const catalogCategories = new Set(
      this.catalog.map(catalogItem => catalogItem.category)
    );
    return Array.from(catalogCategories);
  }

  /**
   * Esporta il catalog in formato JSON
   * @returns {string} Catalog in formato JSON
   */
  exportCatalog() {
    const catalogExport = {
      catalogId: this.catalogId,
      catalogName: this.catalogName,
      catalogVersion: this.catalogVersion,
      catalogItems: this.catalog,
      catalogExportDate: new Date().toISOString()
    };
    return JSON.stringify(catalogExport, null, 2);
  }

  /**
   * Importa items nel catalog da JSON
   * @param {string} catalogJson - JSON del catalog da importare
   */
  importCatalog(catalogJson) {
    try {
      const catalogData = JSON.parse(catalogJson);
      if (catalogData.catalogItems && Array.isArray(catalogData.catalogItems)) {
        catalogData.catalogItems.forEach(catalogItem => {
          this.addToCatalog(catalogItem);
        });
        console.log(`Importati ${catalogData.catalogItems.length} items nel catalog`);
      }
    } catch (catalogError) {
      console.error('Errore durante l\'import del catalog:', catalogError);
    }
  }
}

// Inizializzazione del catalog demo
const mainCatalog = new Catalog();

// Aggiungiamo alcuni items al catalog per la demo
const catalogDemoItems = [
  { name: 'Laptop', price: 999.99, category: 'Electronics' },
  { name: 'Mouse', price: 29.99, category: 'Electronics' },
  { name: 'Desk', price: 299.99, category: 'Furniture' },
  { name: 'Chair', price: 199.99, category: 'Furniture' },
  { name: 'Monitor', price: 449.99, category: 'Electronics' }
];

// Popoliamo il catalog con gli items demo
catalogDemoItems.forEach(catalogItem => {
  mainCatalog.addToCatalog(catalogItem);
});

// Mostriamo le statistiche del catalog
console.log('=== Statistiche del Catalog ===');
console.log(mainCatalog.getCatalogStatistics());

// Cerchiamo nel catalog per categoria
console.log('\n=== Ricerca nel Catalog ===');
const catalogElectronics = mainCatalog.searchCatalogByCategory('Electronics');
console.log(`Trovati ${catalogElectronics.length} items Electronics nel catalog`);

// Export del catalog
console.log('\n=== Export del Catalog ===');
const catalogExportData = mainCatalog.exportCatalog();
console.log('Catalog esportato con successo');
