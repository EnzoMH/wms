from dotenv import load_dotenv
from fastapi import FastAPI, Deponeds
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRouter


from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from loguru import logger
from contextlib import asynccontextmanager

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("비젼스페이스, WMS(Warehouse Management System, 창고관리시스템) RAG API Lifespan 시작")
    try:
        yield
    except Exception as e:
        logger.error(f"비젼스페이스, WMS(Warehouse Management System, 창고관리시스템) RAG API Lifespan 오류: {e}")
        raise e
    logger.info("비젼스페이스, WMS(Warehouse Management System, 창고관리시스템) RAG API Lifespan 종료")

app = FastAPI(
    title = "비젼스페이스, WMS(Warehouse Management System, 창고관리시스템) RAG API",
    description = "비젼스페이스, WMS(Warehouse Management System, 창고관리시스템) RAG API",
    version = "1.0.0",
    lifespan= lifespan
    )

app = CORSMiddleware(
    title = "비젼스페이스, WMS(Warehouse Management System, 창고관리시스템) RAG API",
    description = "비젼스페이스, WMS(Warehouse Management System, 창고관리시스템) RAG API",
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.add_middleware(CORSMiddleware)

# health Chek 함수
@app.get("/health")
async def health_get():
    return {"Message": "Okay"}

@app.get("/health")
async def health_post():
    return {"Message": "Okay"}

# 라우터 포함
app.include_router(chat_router)
app.include_router(search_router)
app.include_router(langchain_router)


if __name__ == "__main__":
	import uvicorn

	print("api docs: http://localhost:8001/docs")
	print("비젼스페이스, WMS(Warehouse Management System, 창고관리시스템) RAG API 시작")
	uvicorn.run(
		app, 
		host="0.0.0.0", 
		port=8001,
		reload=True,
	    )
	print("비젼스페이스, WMS(Warehouse Management System, 창고관리시스템) RAG API 종료")
