import heapq
import numpy as np

TOP_K = 4


def cosine_similarity(a, b):
    return np.dot(a, b)


def find_relevant(query_embedding, doc_embeddings, top_k=TOP_K):
    scores = []

    for i, embedding in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, embedding)
        scores.append((i, float(score)))

    return heapq.nlargest(
        top_k,
        scores,
        key=lambda x: x[1]
    )


def bm25_search(query, bm25, top_k):
    tokens = query.lower().split()

    scores = bm25.get_scores(tokens)

    return heapq.nlargest(
        top_k,
        enumerate(scores),
        key=lambda x: x[1]
    )


def reciprocal_rank_fusion(*rankings, k=60):
    fused = {}

    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0) + 1 / (k + rank + 1)

    return sorted(
        fused.items(),
        key=lambda x: x[1],
        reverse=True
    )


from pypdf import PdfReader 
import docx 
import pandas as pd
import openpyxl 


def character_chunk_streaming( 
    text_blocks,
    chunk_size: int = 2000,
    overlap: int = 400,
) -> list[str]:

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = [] 
    buffer = ""


    for block in text_blocks: 
        if not block:
            continue
            
        buffer += block.strip() + "\n"

        
        while len(buffer) >= chunk_size: 
            chunks.append(buffer[:chunk_size])
            buffer = buffer[chunk_size - overlap:]

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks 

def extract_pdf(path: str):
   
    reader = PdfReader(path)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            yield text 


def extract_docx(path: str): 
    
    doc = docx.Document(path) 
    for p in doc.paragraphs: 
        if p.text.strip(): 
            yield p.text 


def extract_txt(path: str): 
    
    with open(path, "r", encoding="utf-8", errors="replace") as f: 
        for line in f: 
            yield line 


def _row_to_text(row, columns) -> str:
    return "\n".join(f"{col}: {row[col]}" for col in columns)


def extract_csv_chunks(
    path: str,
    batch_size: int = 100000,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:

    def row_blocks():
        for df in pd.read_csv(path, encoding="utf-8-sig", chunksize=batch_size):
            columns = list(df.columns)
            # itertuples is much faster than iterrows for large frames.
            for row in df.itertuples(index=False, name=None):
                yield _row_to_text(dict(zip(columns, row)), columns)

    return character_chunk_streaming(
    row_blocks(),
    chunk_size=chunk_size,
    overlap=overlap,
)


def extract_excel_chunks(
    path: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    

    def row_blocks():
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb.active
            rows_iter = ws.iter_rows(values_only=True)
            header = next(rows_iter, None)
            if header is None:
                return
            columns = [str(c) if c is not None else "" for c in header]
            for row in rows_iter:
                row_dict = dict(zip(columns, row))
                yield _row_to_text(row_dict, columns)
        finally:
            wb.close()

    return character_chunk_streaming(
    row_blocks(),
    chunk_size=chunk_size,
    overlap=overlap,
)


def extract_and_chunk(
    filename: str,
    path: str,
    chunk_size: int = 2000,
    overlap: int = 400,
) -> list[str]:

    name = filename.lower()

    if name.endswith(".pdf"):
        return character_chunk_streaming(
            extract_pdf(path),
            chunk_size,
            overlap,
        )

    elif name.endswith(".docx"):
        return character_chunk_streaming(
            extract_docx(path),
            chunk_size,
            overlap,
        )

    elif name.endswith(".txt"):
        return character_chunk_streaming(
            extract_txt(path),
            chunk_size,
            overlap,
        )

    elif name.endswith(".csv"):
        return extract_csv_chunks(
            path,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    elif name.endswith(".xlsx"):
        return extract_excel_chunks(
            path,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    raise ValueError(f"Unsupported file type: {filename}")
