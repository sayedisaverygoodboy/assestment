from dotenv import load_dotenv
import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import statistics

# NLP / embedding / retrieval / llm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.llms import Ollama
# using groq inference
# from langchain_groq import ChatGroq
# using mistral
from langchain_mistralai import ChatMistralAI



from langchain_core.prompts import PromptTemplate

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk import word_tokenize
import nltk

# load environment variables from .env file
load_dotenv()



nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

# PDF handling
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# plotting
import matplotlib.pyplot as plt

# smoothing for BLEU
bleu_smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


# -------------------- Utilities --------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def pdf_to_text(pdf_path: str) -> str:
    """Simple PDF -> text converter using PyPDF2."""
    if PdfReader is None:
        raise RuntimeError("PyPDF2 not installed. Install via `pip install PyPDF2` to parse PDFs.")
    reader = PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def prepare_corpus(corpus_dir: str) -> Dict[str, str]:
    """
    Load all .txt files from corpus_dir. If PDFs exist, convert them to text
    and save .txt versions.
    Returns dict: filename -> text
    """
    texts = {}
    p = Path(corpus_dir)
    if not p.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    for f in sorted(p.iterdir()):
        if f.suffix.lower() == '.txt':
            texts[f.name] = f.read_text(encoding='utf-8')
        elif f.suffix.lower() == '.pdf':
            txt = pdf_to_text(str(f))
            txt_name = f.stem + '.txt'
            (p / txt_name).write_text(txt, encoding='utf-8')
            texts[txt_name] = txt
    return texts


def smart_chunk_text(text: str, min_chars: int, max_chars: int, overlap: int = 0) -> List[str]:
    """
    Chunk text into windows of size up to max_chars with a minimum of min_chars.
    Attempt to split at sentence boundaries and allow overlap in characters.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""
    for sent in sentences:
        if not sent:
            continue
        if len(current) + len(sent) <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if len(current) >= min_chars:
                chunks.append(current)
                # prepare next chunk with overlap
                if overlap > 0:
                    # take last `overlap` chars of current as prefix for next chunk
                    prefix = current[-overlap:]
                else:
                    prefix = ""
                current = (prefix + " " + sent).strip()
            else:
                # current < min_chars but next sentence pushes over max_chars -> force split
                # attempt to extend until >= min_chars
                current = (current + " " + sent).strip()
                if len(current) >= min_chars:
                    chunks.append(current)
                    current = ""
    if current:
        chunks.append(current)
    # final pass: ensure no chunk > max_chars (split long ones)
    final_chunks = []
    for ch in chunks:
        if len(ch) <= max_chars:
            final_chunks.append(ch)
        else:
            # naive split by chars
            for i in range(0, len(ch), max_chars):
                final_chunks.append(ch[i:i + max_chars])
    return final_chunks


# -------------------- Metrics --------------------
def retrieval_hit_rate(retrieved_sources: List[str], gt_sources: List[str]) -> int:
    return 1 if any(s in gt_sources for s in retrieved_sources) else 0


def retrieval_mrr(retrieved_sources: List[str], gt_sources: List[str]) -> float:
    for idx, s in enumerate(retrieved_sources, start=1):
        if s in gt_sources:
            return 1.0 / idx
    return 0.0


def precision_at_k(retrieved_sources: List[str], gt_sources: List[str], k: int) -> float:
    topk = retrieved_sources[:k]
    hits = sum(1 for s in topk if s in gt_sources)
    return hits / k


def compute_rouge_l(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    r = scorer.score(ref, pred)
    return r['rougeL'].fmeasure


def compute_bleu(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    ref_toks = [word_tokenize(ref)]
    pred_toks = word_tokenize(pred)
    try:
        return sentence_bleu(ref_toks, pred_toks, smoothing_function=bleu_smooth)
    except Exception:
        return 0.0


def compute_cosine_sim(st_model: SentenceTransformer, pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    emb = st_model.encode([pred, ref])
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])


def compute_relevance(pred: str, ref: str) -> float:
    if not ref:
        return 0.0
    p = set(word_tokenize(pred.lower()))
    r = set(word_tokenize(ref.lower()))
    if not r:
        return 0.0
    return len(p & r) / len(r)


def compute_faithfulness(pred: str, retrieved_context: str) -> float:
    if not pred or not retrieved_context:
        return 0.0
    p = set(word_tokenize(pred.lower()))
    s = set(word_tokenize(retrieved_context.lower()))
    if not p:
        return 0.0
    return len(p & s) / len(p)


# -------------------- Indexing --------------------
def index_chunks(chunks: List[Dict[str, Any]], embeddings_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 persist_directory: str = None) -> Chroma:
    hf = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    texts = [c['text'] for c in chunks]
    metadatas = [{'source': c['source'], 'chunk_id': c['id']} for c in chunks]
    vectordb = Chroma.from_texts(texts, hf, metadatas=metadatas, persist_directory=persist_directory)
    return vectordb


# -------------------- LLM + Answer Generation --------------------


# def generate_answer(llm: Ollama, question: str, retrieved_texts: List[str]) -> Tuple[str, List[str]]:
#     """
#     Return (answer, extracted_sources).
#     The prompt asks the model to append a SOURCES: line listing sources (we'll try to extract them).
#     """
#     prompt = """You are an assistant that has access to retrieved passages. Answer the question concisely in one paragraph.
#     Then on a new line write: SOURCES: <comma-separated list of source filenames used>

#     Question: {question}

#     Passages:
#     {passages}

#     Answer:
#     """
#     p = PromptTemplate(template=prompt, input_variables=['question', 'passages'])
#     formatted = p.format(question=question, passages="\n\n".join(retrieved_texts))
    
#     # generate expects a list of 
#     raw_obj = llm.generate([formatted]) 

#     # Extract the text
#     raw = raw_obj.generations[0][0].text  # get the first generated string

#     # attempt to extract SOURCES line
#     sources = []
#     for line in raw.splitlines()[::-1]:
#         if line.strip().lower().startswith('sources:'):
#             src_line = line.split(':', 1)[1]
#             # split by comma or semicolon
#             pieces = re.split(r'[,;]+', src_line)
#             sources = [s.strip() for s in pieces if s.strip()]
#             break
#     return raw.strip(), sources
# def generate_answer(llm: ChatGroq, question: str, retrieved_texts: List[str]) -> Tuple[str, List[str]]:
#     """
#     Return (answer, extracted_sources) using Groq LLM.
#     """
#     prompt_template = """You are an assistant that has access to retrieved passages. Answer the question concisely in one paragraph.
#     Then on a new line write: SOURCES: <comma-separated list of source filenames used>

#     Question: {question}

#     Passages:
#     {passages}

#     Answer:
#     """
#     formatted = prompt_template.format(question=question, passages="\n\n".join(retrieved_texts))

#     # Groq uses messages: list of tuples (role, content)
#     messages = [
#         ("system", "You are a helpful assistant."),
#         ("user", formatted)
#     ]

#     resp = llm.invoke(messages)
#     answer_text = resp.content.strip()

#     # extract SOURCES
#     sources = []
#     for line in answer_text.splitlines()[::-1]:
#         if line.strip().lower().startswith("sources:"):
#             src_line = line.split(":", 1)[1]
#             pieces = re.split(r"[,;]+", src_line)
#             sources = [s.strip() for s in pieces if s.strip()]
#             break

#     return answer_text, sources

def generate_answer(llm: ChatMistralAI, question: str, retrieved_texts: List[str]) -> Tuple[str, List[str]]:
    """
    Return (answer, extracted_sources) using Groq LLM.
    """
    prompt_template = """You are an assistant that has access to retrieved passages. Answer the question concisely in one paragraph.
    Then on a new line write: SOURCES: <comma-separated list of source filenames used>

    Question: {question}

    Passages:
    {passages}

    Answer:
    """
    formatted = prompt_template.format(question=question, passages="\n\n".join(retrieved_texts))

    # Groq uses messages: list of tuples (role, content)
    messages = [
        ("system", "You are a helpful assistant."),
        ("user", formatted)
    ]

    resp = llm.invoke(messages)
    answer_text = resp.content.strip()

    # extract SOURCES
    sources = []
    for line in answer_text.splitlines()[::-1]:
        if line.strip().lower().startswith("sources:"):
            src_line = line.split(":", 1)[1]
            pieces = re.split(r"[,;]+", src_line)
            sources = [s.strip() for s in pieces if s.strip()]
            break

    return answer_text, sources

# -------------------- Analysis --------------------
def analyze_results(all_results: Dict[str, Any], output_dir: str, top_n_failures: int = 5) -> None:
    """
    Create a human-readable analysis file (results_analysis.md) and plots.
    """
    ensure_dir(output_dir)
    md_lines = []
    md_lines.append(f"# RAG Evaluation Analysis\n")
    md_lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
    md_lines.append("\n## Aggregate comparison\n")

    # For plots: collect metric means per chunk key
    metric_keys = ['hit_rate', 'mrr', 'precision@1', 'precision@5', 'rouge_l', 'bleu', 'cosine_sim', 'relevance', 'faithfulness']
    chunk_keys = list(all_results.keys())
    # gather means
    means = {m: [] for m in metric_keys}
    for ck in chunk_keys:
        agg = all_results[ck]['aggregate']
        for m in metric_keys:
            means[m].append(agg.get(m, {}).get('mean', 0.0))

    # write table
    header = "| Chunk Size | " + " | ".join([k.replace('_', ' ').upper() for k in metric_keys]) + " |"
    sep = "|---" * (len(metric_keys) + 1) + "|"
    md_lines.append(header)
    md_lines.append(sep)
    for i, ck in enumerate(chunk_keys):
        row = f"| {ck} | " + " | ".join(f"{means[m][i]:.3f}" for m in metric_keys) + " |"
        md_lines.append(row)

    # plots - one plot per metric
    plots_dir = Path(output_dir) / "plots"
    ensure_dir(str(plots_dir))
    for m in metric_keys:
        plt.figure(figsize=(8, 4))
        plt.plot(chunk_keys, means[m], marker='o')
        plt.title(m.replace('_', ' ').title())
        plt.xlabel("Chunk Size")
        plt.ylabel("Mean")
        plt.grid(True)
        plt.tight_layout()
        fname = plots_dir / f"{m}.png"
        plt.savefig(fname)
        plt.close()
        md_lines.append(f"\n![{m}]({fname.name})\n")

    # failure mode extraction: worst per-chunk by rouge_l (or hit_rate)
    md_lines.append("\n## Failure Mode Analysis\n")
    for ck in chunk_keys:
        per_q = all_results[ck]['per_question']
        # sort by rouge_l ascending (worst first) for answerable questions
        worst = sorted([q for q in per_q if q.get('answerable', True)], key=lambda x: x['metrics'].get('rouge_l', 0.0))
        md_lines.append(f"\n### Chunk: {ck} — Worst {top_n_failures} by ROUGE-L\n")
        for w in worst[:top_n_failures]:
            md_lines.append(f"- ID: {w['id']} — Q: {w['question'][:120]}...\n")
            md_lines.append(f"  - ROUGE-L: {w['metrics']['rouge_l']:.3f}, BLEU: {w['metrics']['bleu']:.3f}, CosSim: {w['metrics']['cosine_sim']:.3f}\n")
            md_lines.append(f"  - Retrieved: {w['retrieved_sources']}\n")
            short_ans = (w['answer'][:250] + '...') if len(w['answer']) > 250 else w['answer']
            md_lines.append(f"  - Answer (truncated): `{short_ans}`\n")

    # recommendation heuristics: simple rule-based
    # choose chunk with highest mean(relevance + faithfulness + hit_rate)
    scores = {}
    for ck in chunk_keys:
        agg = all_results[ck]['aggregate']
        score = (agg['relevance']['mean'] + agg['faithfulness']['mean'] + agg['hit_rate']['mean']) / 3.0
        scores[ck] = score
    best_ck = max(scores.items(), key=lambda x: x[1])[0]
    md_lines.append(f"\n## Recommendation\nBased on a combined heuristic (relevance + faithfulness + hit_rate) the recommended chunk configuration is **{best_ck}**.\n")

    # save analysis markdown
    analysis_path = Path(output_dir) / "results_analysis.md"
    analysis_path.write_text("\n".join(md_lines), encoding='utf-8')
    print(f"Saved analysis to {analysis_path}")


# -------------------- Main evaluation driver --------------------
def evaluate_all(corpus_dir: str, test_dataset_path: str, output_dir: str,
                 chunk_ranges: List[Tuple[int, int]], overlap: int = 0, top_k: int = 5,
                 embeddings_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 persist_directory: str = None):
    ensure_dir(output_dir)
    texts = prepare_corpus(corpus_dir)
    with open(test_dataset_path, 'r', encoding='utf-8') as f:
        test = json.load(f)
    questions = test['test_questions']

    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    # llm = Ollama(model="mistral-7b", verbose=False)
    # mistral-7b is not avilable in mistral's cloud  
    llm = ChatMistralAI(model_name="mistral-large-2512", temperature=0.0)
    

    all_results = {}

    for (min_c, max_c) in chunk_ranges:
        key = f"{min_c}_{max_c}"
        print(f"\n--- Evaluating chunks {min_c}-{max_c} (overlap={overlap}) ---")
        # build chunks
        chunks = []
        cid = 0
        for fname, text in texts.items():
            pieces = smart_chunk_text(text, min_c, max_c, overlap=overlap)
            for ptxt in pieces:
                chunks.append({'id': f"{fname}#c{cid}", 'text': ptxt, 'source': fname})
                cid += 1

        vectordb = index_chunks(chunks, embeddings_model_name=embeddings_model_name, persist_directory=persist_directory)

        per_q_results = []
        for q in questions:
            qid = q['id']
            qtext = q['question']
            ground = q['ground_truth']
            gt_docs = q.get('source_documents', [])

            docs = vectordb.similarity_search_with_score(qtext, k=top_k)
            retrieved_texts = [d.page_content for d, score in docs]
            retrieved_meta = [d.metadata.get('source') for d, score in docs]

            # retrieval metrics
            hit = retrieval_hit_rate(retrieved_meta, gt_docs)
            mrr = retrieval_mrr(retrieved_meta, gt_docs)
            p1 = precision_at_k(retrieved_meta, gt_docs, 1)
            p5 = precision_at_k(retrieved_meta, gt_docs, top_k)

            # LLM answer + extracted sources
            answer, extracted_sources = generate_answer(llm, qtext, retrieved_texts[:3])

            # answer metrics
            rouge_l = compute_rouge_l(answer, ground) if q.get('answerable', True) else 0.0
            bleu = compute_bleu(answer, ground) if q.get('answerable', True) else 0.0
            cos_sim = compute_cosine_sim(st_model, answer, ground) if q.get('answerable', True) else 0.0
            relevance = compute_relevance(answer, ground) if q.get('answerable', True) else 0.0
            faith = compute_faithfulness(answer, " ".join(retrieved_texts))

            per_q_results.append({
                'id': qid,
                'question': qtext,
                'retrieved_sources': retrieved_meta,
                'extracted_sources': extracted_sources,
                'answer': answer,
                'metrics': {
                    'hit_rate': hit,
                    'mrr': mrr,
                    'precision@1': p1,
                    f'precision@{top_k}': p5,
                    'rouge_l': rouge_l,
                    'bleu': bleu,
                    'cosine_sim': cos_sim,
                    'relevance': relevance,
                    'faithfulness': faith
                },
                'answerable': q.get('answerable', True)
            })

        # aggregate
        metric_names = ['hit_rate', 'mrr', 'precision@1', f'precision@{top_k}', 'rouge_l', 'bleu', 'cosine_sim', 'relevance', 'faithfulness']
        agg = {}
        for m in metric_names:
            vals = [q['metrics'].get(m, 0.0) for q in per_q_results]
            if vals:
                agg[m] = {'mean': statistics.mean(vals), 'median': statistics.median(vals), 'stdev': statistics.pstdev(vals) if len(vals) > 1 else 0.0}
            else:
                agg[m] = {'mean': 0.0, 'median': 0.0, 'stdev': 0.0}

        all_results[key] = {'per_question': per_q_results, 'aggregate': agg}

        # save per-chunk intermediate
        out_path = Path(output_dir) / f"results_{key}.json"
        out_path.write_text(json.dumps(all_results[key], indent=2), encoding='utf-8')
        print(f"Saved intermediate results to {out_path}")

    # final save
    final_path = Path(output_dir) / "test_results.json"
    final_path.write_text(json.dumps(all_results, indent=2), encoding='utf-8')
    print(f"\nSaved final results to {final_path}")

    # run analysis
    analyze_results(all_results, output_dir)

# --------------------- Static --------------------
CORPUS_DIR = "./corpus"
TEST_DATASET = "./test_dataset.json"
OUTPUT_DIR = "./results"

CHUNK_RANGES = [
    (200, 300),
    (500, 600),
    (800, 1000)
]

OVERLAP = 0
TOP_K = 5
PERSIST_DIR = None  # set to "./chroma_store" if you want persistence

# -------------------- RUN EVALUATION --------------------

if __name__ == "__main__":
    evaluate_all(
        corpus_dir=CORPUS_DIR,
        test_dataset_path=TEST_DATASET,
        output_dir=OUTPUT_DIR,
        chunk_ranges=CHUNK_RANGES,
        overlap=OVERLAP,
        top_k=TOP_K,
        persist_directory=PERSIST_DIR
    )
# -------------------- CLI --------------------
# if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--corpus_dir', type=str, required=True)
    # parser.add_argument('--test_dataset', type=str, required=True)
    # parser.add_argument('--output_dir', type=str, default='./results')
    # parser.add_argument('--chunk_ranges', type=str, nargs='+', default=['200-300', '500-600', '800-1000'],
    #                     help="Space separated ranges like 200-300 500-600")
    # parser.add_argument('--overlap', type=int, default=0, help="Character overlap between chunks")
    # parser.add_argument('--top_k', type=int, default=5)
    # parser.add_argument('--persist', type=str, default=None, help="Chroma persistence directory (optional)")
    # args = parser.parse_args()

    # # parse chunk ranges
    # parsed = []
    # for r in args.chunk_ranges:
    #     low, high = r.split('-', 1)
    #     parsed.append((int(low), int(high)))

    # evaluate_all(args.corpus_dir, args.test_dataset, args.output_dir, parsed,
    #              overlap=args.overlap, top_k=args.top_k, persist_directory=args.persist)
