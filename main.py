from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import openai
import os
from googlesearch import search
from langchain.document_loaders import WebBaseLoader
import nest_asyncio
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document

api_key = os.environ['OPENAI_API_KEY']
openai.api_key = api_key
embeddings = OpenAIEmbeddings()

def create_db_from_urls(urls, query):
    nest_asyncio.apply() ## accelerate process
    loader = WebBaseLoader(urls)
    docs = loader.aload()

    responses = []
    number = 0
    for doc in docs:
        
        template = """
            Summarise the input document. 
            Only summarise {} related parts. 
            Information about the interview steps and asked questions need to be included in the summary. 
            Articles may include interviews about other companies. Please "Do not" include infomation of other companies.
        """.format(query)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[{"role": "system", "content":template}, 
                    {"role": "user", "content": doc.page_content}],
        )
        responses.append(Document(page_content=str(number)+';;;'+response["choices"][0]["message"]["content"]))
        number += 1

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_splitted = text_splitter.split_documents(responses)
    db = FAISS.from_documents(docs_splitted, embeddings)
    
    return db


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k) # get the first k highest similarity articles
    numbers = [int(doc.page_content.split(";;;")[0]) for doc in docs]
    docs_page_content = " ".join([d.page_content for d in docs])

    
    template=f"""
        You are a helpful assistant that that can answer questions about job interview requirements and questions
        based on the articles.
        
        List the questions that were asked in the {query} interview, and highlight the skills that are required for the role.
        By searching the input articles.

        Do not include the infomation in other companies.
        Only use the factual information from the transcript to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".
        Your answers should be verbose and detailed.
        """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "system", "content":template}, 
                {"role": "user", "content":docs_page_content}],
    )
    result=response["choices"][0]["message"]["content"]
    return result, docs, numbers



# query = "Optiver trader interview questions"
query = "Appier Data Analyst面試心得"
urls = []
titles = []
for result in search(query, num_results=20, sleep_interval=5, advanced=True):
    urls.append(result.url)
    titles.append(result.title)

db = create_db_from_urls(urls, query)
res, docs, numbers = get_response_from_query(db, query, k=10)
print(res)
# print(numbers) # selected articles
print([titles[i] for i in numbers])
print([urls[i]for i in numbers])


