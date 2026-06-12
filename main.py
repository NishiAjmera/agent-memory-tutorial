import os
import uuid
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set in .env")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from google.adk.agents import Agent, SequentialAgent
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.adk.runners import Runner
from google.adk.tools import FunctionTool
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Shared state ──────────────────────────────────────────────────────────────

MODEL = "gemini-3.5-flash"

# Level 1: In-memory session service, sessions keyed by session_id
l1_service = InMemorySessionService()
l1_sessions: dict[str, str] = {}  # user_key -> session_id

# Level 2: Multi-agent state, sessions keyed by session_id
l2_service = InMemorySessionService()
l2_sessions: dict[str, str] = {}

# Level 3: Persistent (SQLite)
DB_PATH = "memory_demo.db"
l3_service = DatabaseSessionService(db_url=f"sqlite:///{DB_PATH}")
l3_sessions: dict[str, str] = {}

# Level 4: Callbacks
l4_service = InMemorySessionService()
l4_sessions: dict[str, str] = {}

# Level 5: Custom tools (in-process dict as "DB")
l5_service = InMemorySessionService()
l5_sessions: dict[str, str] = {}
user_preferences_store: dict[str, dict] = {}

APP_NAME = "memory-walkthrough"


# ── Helpers ───────────────────────────────────────────────────────────────────

async def ensure_session(service, sessions: dict, key: str, state: dict | None = None) -> str:
    if key not in sessions:
        session = await service.create_session(
            app_name=APP_NAME,
            user_id=key,
            state=state or {},
        )
        sessions[key] = session.id
    return sessions[key]


async def run_agent(runner: Runner, user_id: str, session_id: str, message: str) -> str:
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=message)])
    final = ""
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            final = event.content.parts[0].text
    return final or "(no response)"


# ── Level 1: Session & State ──────────────────────────────────────────────────

l1_agent = Agent(
    name="level1_agent",
    model=MODEL,
    instruction="You are a helpful assistant. Remember everything the user tells you in this conversation.",
)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "demo_user"


@app.post("/api/level1/chat")
async def level1_chat(req: ChatRequest):
    session_id = await ensure_session(l1_service, l1_sessions, req.user_id)
    runner = Runner(agent=l1_agent, app_name=APP_NAME, session_service=l1_service)
    reply = await run_agent(runner, req.user_id, session_id, req.message)
    return {"reply": reply, "session_id": session_id}


@app.post("/api/level1/reset")
async def level1_reset(req: ChatRequest):
    l1_sessions.pop(req.user_id, None)
    return {"status": "reset"}


# ── Level 2: Multi-Agent State ────────────────────────────────────────────────

researcher = Agent(
    name="researcher",
    model=MODEL,
    instruction="Extract the key facts from the user message and summarize them in 2-3 bullet points.",
    output_key="research_notes",
)

synthesizer = Agent(
    name="synthesizer",
    model=MODEL,
    instruction=(
        "You have these research notes: {research_notes}\n"
        "Now write a friendly, conversational response to the user using those notes."
    ),
)

l2_pipeline = SequentialAgent(name="l2_pipeline", sub_agents=[researcher, synthesizer])


@app.post("/api/level2/chat")
async def level2_chat(req: ChatRequest):
    session_id = await ensure_session(l2_service, l2_sessions, req.user_id)
    runner = Runner(agent=l2_pipeline, app_name=APP_NAME, session_service=l2_service)
    reply = await run_agent(runner, req.user_id, session_id, req.message)

    session = await l2_service.get_session(app_name=APP_NAME, user_id=req.user_id, session_id=session_id)
    notes = session.state.get("research_notes", "")
    return {"reply": reply, "shared_state": {"research_notes": notes}}


@app.post("/api/level2/reset")
async def level2_reset(req: ChatRequest):
    l2_sessions.pop(req.user_id, None)
    return {"status": "reset"}


# ── Level 3: Persistence ──────────────────────────────────────────────────────

l3_agent = Agent(
    name="level3_agent",
    model=MODEL,
    instruction=(
        "You are a long-term memory assistant. "
        "Always reference things the user told you in past conversations when relevant."
    ),
)


@app.post("/api/level3/chat")
async def level3_chat(req: ChatRequest):
    session_id = await ensure_session(l3_service, l3_sessions, req.user_id)
    runner = Runner(agent=l3_agent, app_name=APP_NAME, session_service=l3_service)
    reply = await run_agent(runner, req.user_id, session_id, req.message)
    return {"reply": reply, "session_id": session_id, "db_file": DB_PATH}


@app.post("/api/level3/reset")
async def level3_reset(req: ChatRequest):
    # Drop from our mapping but keep DB entry to show persistence
    l3_sessions.pop(req.user_id, None)
    return {"status": "reset (DB record kept — restart the app and it will still remember)"}


# ── Level 4: Callbacks ────────────────────────────────────────────────────────

interaction_log: list[str] = []


def after_agent_callback(callback_ctx: CallbackContext) -> genai_types.Content | None:
    """Log every agent turn automatically via callback."""
    if callback_ctx.agent_name and callback_ctx.agent_name != "__root__":
        msg = f"[callback] Agent '{callback_ctx.agent_name}' finished a turn."
        interaction_log.append(msg)
    return None


l4_agent = Agent(
    name="level4_agent",
    model=MODEL,
    instruction="You are a helpful assistant. Answer concisely.",
    after_agent_callback=after_agent_callback,
)


@app.post("/api/level4/chat")
async def level4_chat(req: ChatRequest):
    session_id = await ensure_session(l4_service, l4_sessions, req.user_id)
    runner = Runner(agent=l4_agent, app_name=APP_NAME, session_service=l4_service)
    reply = await run_agent(runner, req.user_id, session_id, req.message)
    return {"reply": reply, "callback_log": list(interaction_log[-5:])}


@app.post("/api/level4/reset")
async def level4_reset(req: ChatRequest):
    l4_sessions.pop(req.user_id, None)
    interaction_log.clear()
    return {"status": "reset"}


# ── Level 5: Custom Tools ─────────────────────────────────────────────────────

def recall_user_preferences(user_id: str, tool_context: ToolContext) -> dict:
    """Read the user's stored preferences from the memory store."""
    prefs = user_preferences_store.get(user_id, {})
    tool_context.state["recalled_prefs"] = json.dumps(prefs)
    return {"preferences": prefs, "found": bool(prefs)}


def save_user_preferences(user_id: str, preferences: dict, tool_context: ToolContext) -> dict:
    """Save the user's preferences to the memory store."""
    user_preferences_store.setdefault(user_id, {}).update(preferences)
    tool_context.state["saved_prefs"] = json.dumps(user_preferences_store[user_id])
    return {"saved": True, "stored": user_preferences_store[user_id]}


recall_tool = FunctionTool(func=recall_user_preferences)
save_tool = FunctionTool(func=save_user_preferences)

l5_agent = Agent(
    name="level5_agent",
    model=MODEL,
    instruction=(
        "You are a personal assistant with memory tools. "
        "Use recall_user_preferences to check what you know about the user before answering. "
        "Use save_user_preferences when the user shares personal info (name, likes, preferences). "
        "Always pass the user_id '{user_id}' to tools."
    ),
    tools=[recall_tool, save_tool],
)


@app.post("/api/level5/chat")
async def level5_chat(req: ChatRequest):
    session_id = await ensure_session(l5_service, l5_sessions, req.user_id)
    # Inject user_id into instruction at runtime
    agent = Agent(
        name="level5_agent",
        model=MODEL,
        instruction=(
            f"You are a personal assistant with memory tools. "
            f"Use recall_user_preferences to check what you know about the user. "
            f"Use save_user_preferences when user shares personal info. "
            f"Always pass user_id='{req.user_id}' to the tools."
        ),
        tools=[recall_tool, save_tool],
    )
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=l5_service)
    reply = await run_agent(runner, req.user_id, session_id, req.message)
    return {
        "reply": reply,
        "memory_store": user_preferences_store.get(req.user_id, {}),
    }


@app.post("/api/level5/reset")
async def level5_reset(req: ChatRequest):
    l5_sessions.pop(req.user_id, None)
    user_preferences_store.pop(req.user_id, None)
    return {"status": "reset"}


# ── Level 6: Multimodal Memory (concept demo) ─────────────────────────────────

multimodal_store: list[dict] = []

l6_agent = Agent(
    name="level6_agent",
    model=MODEL,
    instruction=(
        "You are a multimodal memory assistant. "
        "When the user describes something (an image, video, or document), store the description as a memory entry. "
        "When answering questions, first recall any relevant past memories and reference them explicitly."
    ),
)

l6_service = InMemorySessionService()
l6_sessions: dict[str, str] = {}


@app.post("/api/level6/chat")
async def level6_chat(req: ChatRequest):
    session_id = await ensure_session(l6_service, l6_sessions, req.user_id)

    # Prepend any stored memories as context
    memory_context = ""
    if multimodal_store:
        entries = "\n".join(f"- [{e['type']}] {e['content']}" for e in multimodal_store[-5:])
        memory_context = f"Past multimodal memories:\n{entries}\n\n"

    augmented_msg = memory_context + req.message

    runner = Runner(agent=l6_agent, app_name=APP_NAME, session_service=l6_service)
    reply = await run_agent(runner, req.user_id, session_id, augmented_msg)

    # Auto-store if user is describing something
    if any(kw in req.message.lower() for kw in ["image", "photo", "video", "document", "file", "picture"]):
        multimodal_store.append({"type": "auto-detected", "content": req.message})

    return {"reply": reply, "memory_bank": multimodal_store[-5:]}


@app.post("/api/level6/store")
async def level6_store(req: ChatRequest):
    multimodal_store.append({"type": "text", "content": req.message})
    return {"stored": True, "bank_size": len(multimodal_store)}


@app.post("/api/level6/reset")
async def level6_reset(req: ChatRequest):
    l6_sessions.pop(req.user_id, None)
    multimodal_store.clear()
    return {"status": "reset"}


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
