#read and aggregate all *.md and *.ipynb files recursively from root folder
from langchain_core.documents import Document
import os, base64
import json
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
from markitdown import MarkItDown
from openai import OpenAI
import sys
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
load_dotenv('../.env')

root_dir = Path(os.getcwd())
VISION_PROVIDER= "openai" #ollama, openai
VISION_MODEL = "gpt-4o" #deepseek-ocr:3b, qwen3-vl:8b, gpt-4o
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B" #google/embeddinggemma-300m"

def parse_args():
    """
    Parse command line arguments to override default configuration parameters.

    <sample>
    --vision_provider=ollama --vision_model=qwen3-vl:8b --embedding_model=Qwen/Qwen3-Embedding-0.6B \n
    --vision_provider=openai --vision_model=gpt-4o --embedding_model=google/embeddinggemma-300m
    </sample>
    """
    params = ["VISION_PROVIDER","VISION_MODEL","EMBEDDING_MODEL"]
    for arg in sys.argv:
        for param in params:
            if arg.lower().startswith(f"--{param.lower()}="):
                _v = arg.split("=", 1)[1]
                globals()[param] = _v

def load_documents(file_path: Path) -> list[Document]:
    """Load a JSONL file and return a list of Documents"""
    if not file_path.exists():
        return []
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            doc = Document(
                page_content=record['page_content'],
                metadata=record['metadata']
            )
            documents.append(doc)
    return documents

def should_include(file_path: Path) -> bool:
    """Check if file should be included based on path rules"""
    parts = file_path.relative_to(root_dir).parts
    for part in parts:
        # patterns for exclusion
        startswith = {'.'}
        exact_match = {'tmp', 'static', '__pycache__'}
        contains = {'.ignore'}        
        if (part[0] in startswith or 
            part in exact_match or 
            any(pattern in part for pattern in contains)):
            return False
    return True

def process_notebook(ipynb_files: list[Path], splitter: MarkdownHeaderTextSplitter) -> list[Document]:
    """Process notebook content and split into documents"""
    def sanitize_attachments(notebook_path: Path):
        """
        Process a single notebook: extract images, save them, update markdown references, and remove attachments.
        """    
        def save_image(image_data, output_path):
            """
            Decode base64 image data and save to file.
            """
            # Remove data URL prefix if present (e.g., "data:image/png;base64,")
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]            
            image_bytes = base64.b64decode(image_data)            
            with open(output_path, 'wb') as image_file:
                image_file.write(image_bytes)        

        notebook_dir = os.path.dirname(notebook_path)        
        with open(notebook_path, 'r', encoding='utf-8') as file:
            notebook_content = json.load(file)        
        modified = False 
        for cell in notebook_content.get('cells', []):
                if cell.get('cell_type') == 'markdown' and 'attachments' in cell:
                    cell_id = cell.get('id', 'unknown')
                    attachments = cell['attachments']                    
                    # Process each attachment
                    for attachment_name, attachment_data in attachments.items():
                        # Determine the image format
                        image_format = None
                        image_data = None                        
                        if 'image/png' in attachment_data:
                            image_format = 'png'
                            image_data = attachment_data['image/png']
                        elif 'image/jpeg' in attachment_data:
                            image_format = 'jpg'
                            image_data = attachment_data['image/jpeg']
                        elif 'image/gif' in attachment_data:
                            image_format = 'gif'
                            image_data = attachment_data['image/gif']                        
                        if image_data:
                            # Create new filename with cell_id prefix
                            original_ext = Path(attachment_name).suffix or f'.{image_format}'
                            new_filename = f"{cell_id}_{attachment_name}"                            
                            # Save image to disk
                            image_path = os.path.join(notebook_dir, new_filename)
                            save_image(image_data, image_path)
                            
                            # Update markdown references in cell source
                            if isinstance(cell['source'], list):
                                cell['source'] = [
                                    line.replace(f'attachment:{attachment_name}', f'./{new_filename}')
                                    for line in cell['source']
                                ]
                            else:
                                cell['source'] = cell['source'].replace(
                                    f'attachment:{attachment_name}', 
                                    f'./{new_filename}'
                                )                            
                            modified = True                    
                    # Remove attachments node from cell
                    del cell['attachments']            
        # Save modified notebook
        if modified:
            with open(notebook_path, 'w', encoding='utf-8') as file:
                json.dump(notebook_content, file, indent=1, ensure_ascii=False)
            print(f"✓ Image processed for ipynb: {notebook_path}")          
        pass

    def run(notebook_path: Path) -> list[Document]:
        """Parse a Jupyter notebook and extract markdown and code cells"""
        #pre-parse to sanitize attachments
        sanitize_attachments(notebook_path)

        #open
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        docs = []
        cell_number = 0
        
        for cell in nb.get('cells', []):
            cell_type = cell.get('cell_type')
            source = cell.get('source', [])
            
            # Join source lines into a single string
            if isinstance(source, list):
                content = ''.join(source)
            else:
                content = source
            
            # Skip empty cells
            if not content.strip():
                continue
            
            cell_number += 1
            
            if cell_type == 'markdown':
                # Parse markdown cells with header splitter
                try:
                    splits = splitter.split_text(content)
                    for split in splits:
                        split.metadata.update({
                            'source': str(notebook_path),
                            'cell_type': 'markdown',
                            'cell_number': cell_number,
                        })
                        docs.append(split)
                except Exception as e:
                    # If header splitting fails, add as-is
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            'source': str(notebook_path),
                            'cell_type': 'markdown',
                            'cell_number': cell_number,
                        }
                    ))
            
            elif cell_type == 'code':
                # Add code cells with proper formatting
                docs.append(Document(
                    page_content=f"```python\n{content}\n```",
                    metadata={
                        'source': str(notebook_path),
                        'cell_type': 'code',
                        'cell_number': cell_number,
                    }
                ))
        
        return docs    
    
    documents = []
    for nb_file in ipynb_files:
        try:
            nb_docs = run(nb_file)
            documents.extend(nb_docs)
        except Exception as e:
            print(f"  ✗ Error processing {nb_file}: {e}")    
    return documents

def process_markdown(md_files: list[Path], splitter: MarkdownHeaderTextSplitter) -> list[Document]:
    """Process markdown content and split into documents"""
    documents = []
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                splits = splitter.split_text(content)    
                for doc in splits:
                    doc.metadata.update({'source': str(md_file), 'file_type': 'markdown'})
                    documents.append(doc)
        except Exception as e:
            print(f"  ✗ Error processing {md_file}: {e}")
    return documents

def process_images(img_files: list[Path],source_file: Path, splitter: MarkdownHeaderTextSplitter) -> list[Document]:
    """Process image files and extract documents using MarkItDown"""
    documents = []
    if VISION_PROVIDER == "ollama":
        llm_client = OpenAI(base_url="http://localhost:11434/", api_key="sk-dummy-key")
    else:
        llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    img_converter = MarkItDown(llm_client=llm_client, llm_model=VISION_MODEL)    
    try:
        _images = list(filter(lambda d: d.metadata.get('file_type') == 'image', load_documents(source_file)))
        print(f"  - Loaded {_images.__len__()} previously processed images from {source_file}")
    except Exception as e:
        print(f"  ✗ Error loading previously processed images: {e}")
        _images = []
    
    def run(img_file):
        """Process a single image file"""
        _find = [d for d in _images if d.metadata.get('source') == str(img_file)]
        if _find:
            #print(f"  - Skipping already processed image {img_file}")
            return _find
        try:
            _result = img_converter.convert(str(img_file))
            splits = splitter.split_text(_result.text_content)    
            for doc in splits:
                doc.metadata.update({'source': str(img_file), 'file_type': 'image'})
            print(f"  ✓ Processed {img_file}, extracted {len(splits)} documents")
            return splits
        except Exception as e:
            print(f"  ✗ Error processing {img_file}: {e}")
            return []
    
    # Process images in batches of 4
    batch_size = 4
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for i in range(0, len(img_files), batch_size):
            batch = img_files[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} images)...")
            
            # Process batch in parallel
            results = list(executor.map(run, batch))
            
            # Add all documents from this batch
            for splits in results:
                documents.extend(splits)
    return documents

def copy_static_file(app_dir: Path, paths: list[list[Path]]):
    def copy_file(file: Path, dest: Path):
        """Copy a file from src to dest"""
        dest_file = dest / file
        #create parent dirs if not existing
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        if not dest_file.exists():
            with open(file, 'rb') as src_f, open(dest_file, 'wb') as dest_f:
                dest_f.write(src_f.read())

    static_dir = app_dir / 'static'
    static_dir.mkdir(parents=True, exist_ok=True)
    #[img,md files] copy
    for path in paths:
        for file in path:
            copy_file(file, static_dir)

def save_documents(documents: list[Document], output_file: Path):
    """Serialize documents to a JSONL file"""
    with open(output_file, 'w', encoding='utf-8') as f:  # 'w' mode truncates the file
        for doc in documents:
            json.dump({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            }, f)
            f.write('\n')
    print(f"Serialized {len(documents)} documents to {output_file}")        

def create_vector_store(base_path: Path, documents: list[Document]):    
    """Create vector store from knowledge base documents"""
    def faiss():
        from langchain_community.vectorstores import FAISS
        import os, shutil
        _storage_id= base_path / 'faiss'
        if os.path.exists(_storage_id):
            shutil.rmtree(_storage_id, ignore_errors=True)
        _db = FAISS.from_documents(documents, embeddings)
        _db.save_local(str(_storage_id))
        del _db        
        
    def chroma():
        from langchain_community.vectorstores import Chroma
        import os, shutil
        _storage_id= base_path / 'chroma'
        if os.path.exists(_storage_id):
            shutil.rmtree(_storage_id, ignore_errors=True)
        _db = Chroma.from_documents(documents, embeddings, collection_name="default", persist_directory=str(_storage_id))
        del _db    

    import torch
    from langchain_huggingface import HuggingFaceEmbeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(    
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': device}  
)
    print("Creating faiss vector stores...")
    faiss()
    print("Creating chroma vector stores...")
    chroma()

def generate_kb():
    """Generate knowledge base from documents"""
    # get current directory
    current_dir = Path(os.path.dirname(__file__))
    print(f"Generating knowledge base in {current_dir}")
    source_file =  current_dir / 'tmp' / 'app.jsonl'
    output_file =  current_dir / 'tmp' / 'out.jsonl'

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "topic"),
            ("###", "detail"),
        ],
        strip_headers=True,
        return_each_line=False
    )    

    documents = []

    # Process markdown files
    print("Processing markdown files...")
    md_files = [os.path.relpath(f, root_dir) for f in root_dir.rglob('*.md') if should_include(f)]
    documents.extend(process_markdown(md_files, splitter))

    # Process notebook files
    print("Processing notebook files...")
    ipynb_files = [os.path.relpath(f, root_dir) for f in root_dir.rglob('*.ipynb') if should_include(f)]   
    documents.extend(process_notebook(ipynb_files, splitter)) 

    # process images
    img_extensions = {'.png', '.jpg'}
    img_files = [
        os.path.relpath(f, root_dir) for f in root_dir.rglob('*') 
        if f.suffix.lower() in img_extensions and should_include(f)
    ]     
    documents.extend(process_images(img_files, source_file, splitter))

    # copy static files
    copy_static_file(current_dir, [md_files, ipynb_files, img_files])

    # save all documents
    save_documents(documents, output_file)

    # create vector stores
    create_vector_store(current_dir / 'tmp' / 'db', documents)

    print(f"\n{'='*80}")
    print(f"Knowledge base generated in {current_dir}")
    print(f"TOTAL: Loaded {len(documents)} documents")
    print(f"  - From {len(md_files)} markdown files -> {len([d for d in documents if d.metadata.get('file_type', '') == "markdown"])} documents")
    print(f"  - From {len(ipynb_files)} notebook files -> {len([d for d in documents if d.metadata.get('source', '').endswith('.ipynb')])} documents")
    print(f"  - From {len(img_files)} image files -> {len([d for d in documents if d.metadata.get('file_type', '') == "image"])} documents")
    print(f"{'='*80}\n")        

def copy_env_file():
    """Copy .env file from root to app directory"""
    import shutil
    root_env = root_dir / '.env'
    app_env = Path(os.path.dirname(__file__)) / '.env'
    if root_env.exists():
        shutil.copy(root_env, app_env)
        print(f"Copied {root_env} to {app_env}")
if __name__ == "__main__":
    parse_args()
    generate_kb()
    copy_env_file()