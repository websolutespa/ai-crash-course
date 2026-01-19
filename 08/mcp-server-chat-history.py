"""
MCP Server for Chat History API using fastmcp.
Maps all endpoints from chat-api.py as MCP tools and resources.
"""
import os
import datetime as dt
from typing import Generator, Literal, Optional, Any
from contextlib import contextmanager
from fastmcp import FastMCP, Context
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
import argparse
import asyncio

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--transport", type=str, default=None, help="mcp transport: http, stdio")
args = argument_parser.parse_args()

_base_dir = os.path.dirname(os.path.abspath(__file__))
_db_dir = os.path.join(_base_dir, "tmp", "db")
os.makedirs(_db_dir, exist_ok=True)  # ensure sqlite path exists before engine connects
DB_URL = f"sqlite:///{os.path.join(_db_dir, 'chat.db')}"
ANONYMOUS_USER_ID = "anonymous"

class Base(DeclarativeBase):
	pass

def utcnow() -> dt.datetime:
	return dt.datetime.now(dt.timezone.utc)

class Chat(Base):
    __tablename__ = "chats"
    
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(32),index=True, default=ANONYMOUS_USER_ID)
    title: Mapped[str] = mapped_column(String(255), default="New chat")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    messages: Mapped[list["Message"]] = relationship(back_populates="chat",cascade="all, delete-orphan",order_by="Message.created_at")
    feedback: Mapped[Optional["Feedback"]] = relationship(back_populates="chat",uselist=False,cascade="all, delete-orphan")   

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    chat: Mapped[Chat] = relationship(back_populates="messages")


class Feedback(Base):
    __tablename__ = "feedbacks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), unique=True, index=True)
    rating: Mapped[int] = mapped_column(Integer)
    comment: Mapped[str] = mapped_column(Text, default="")
    sentiment: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    chat: Mapped[Chat] = relationship(back_populates="feedback")


engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Eagerly create tables at import time so both FastAPI and MCP entrypoints have schema ready
Base.metadata.create_all(bind=engine)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize FastMCP server

mcp = FastMCP(
    "Chat History MCP Server",
    instructions="MCP server exposing chat history API tools and resources.")

# Database setup
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Ensure tables exist
Base.metadata.create_all(bind=engine)


@contextmanager
def get_db():
    """Database session context manager."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def to_dict(obj) -> dict[str, Any]:
    """Convert SQLAlchemy model to clean dict without internal state."""
    if obj is None:
        return None
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


# ============================================================================
# RESOURCES (Read-only operations exposed as resources)
# ============================================================================

@mcp.resource("chats://list")
def list_chats_resource() -> str:
    """Resource for listing all chats."""
    with get_db() as db:
        chats = db.query(Chat).order_by(Chat.updated_at.desc()).all()
        
        result = f"Total chats: {len(chats)}\n\n"
        for chat in chats:
            result += f"#{chat.id}: {chat.title} (updated: {chat.updated_at})\n"
        
        return result  


# ============================================================================
# TOOLS (All CRUD operations)
# ============================================================================

@mcp.tool()
async def create_chat(
    user_id: Optional[str] = None,
    title: Optional[str] = None,
    ctx: Context = None
) -> dict[str, Any]:
    """
    Create a new chat for a user (defaults to anonymous).
    
    Args:
        user_id: Optional user identifier (max 32 chars)
        title: Optional chat title (max 255 chars)
    
    Returns:
        Dictionary with chat id, title, user_id, created_at, and updated_at
    """
    with get_db() as db:                
        chat = Chat(
             user_id=user_id or ANONYMOUS_USER_ID, 
             title=title or "New chat"
             )
        db.add(chat)
        db.commit()
        db.refresh(chat)
        await ctx.log(f"Created new chat `{chat.id}` for user `{chat.user_id}`")
        return to_dict(chat)

@mcp.tool()
def get_chat(
    chat_id: Optional[int] = None,
    user_id: Optional[str] = None,
    include_messages: bool = False,
    include_feedback: bool = False
) -> list[dict[str, Any]]:
    """
    Retrieve a chat by chat_id or all chats by user_id.
    
    Args:
        chat_id: The chat ID to retrieve (returns array with single chat)
        user_id: The user ID to retrieve chats for (returns all chats for user)
        include_messages: Whether to include chat messages in the response
        include_feedback: Whether to include chat feedback in the response
    
    Returns:
        List of chats metadata with feedback (if requested), and messages (if requested)
        An empty list is returned if no chats found, meaning that the user has no chats yet and
        must create a new one for further interactions.
    """
    if chat_id is None and user_id is None:
        return {"error": "Either chat_id or user_id must be provided."}
    if chat_id is not None:    
         _filter = Chat.id == chat_id
    else: #user
         _filter = Chat.user_id == user_id
    with get_db() as db:
        chats = db.query(Chat).filter(_filter).order_by(Chat.updated_at.desc()).all()        
        result = []
        for chat in chats:
            chat_data = to_dict(chat)
            if include_feedback:
                feedback = db.query(Feedback).filter(Feedback.chat_id == chat.id).first()
                chat_data["feedback"] = to_dict(feedback)
            if include_messages:
                messages = (
                    db.query(Message)
                    .filter(Message.chat_id == chat.id)
                    .order_by(Message.created_at.asc())
                    .all()
                )
                chat_data["messages"] = [to_dict(msg) for msg in messages]            
            result.append(chat_data)        
        return result        

@mcp.tool()
async def update_chat_title(
    chat_id: int,
    title: str,
    ctx: Context = None
) -> dict[str, Any]:
    """
    Update a chat's title and bump its update timestamp.
    
    Args:
        chat_id: The chat ID to update
        title: New title for the chat (optional)
    
    Returns:
        Dictionary with updated chat metadata
    """
    with get_db() as db:
        chat = db.get(Chat, chat_id)
        if not chat:
            return {"error": "Chat not found", "chat_id": chat_id}
        
        if title is not None:
            chat.title = title.strip()        
        chat.updated_at = utcnow()
        db.commit()
        db.refresh(chat)
        await ctx.log(f"Updated title for chat `{chat.id}` to `{chat.title}`")
        return to_dict(chat)

@mcp.tool()
async def delete_chat(chat_id: int, ctx: Context) -> dict[str, Any]:
    """
    Delete a chat and cascade-delete its messages.
    
    Args:
        chat_id: The chat ID to delete
    
    Returns:
        Success message or error
    """
    _elicit = await ctx.elicit(
        message="Are you sure you want to delete this chat? This action cannot be undone.",
        response_type= None,
    )
    if not _elicit.action == "accept":
        return {"error": "Chat deletion cancelled by user."}    
    with get_db() as db:
        chat = db.get(Chat, chat_id)
        if not chat:
            return {"error": "Chat not found", "chat_id": chat_id}
        
        db.delete(chat)
        db.commit()
        await ctx.log(f"Deleted chat `{chat_id}`")
        return {"success": True, "message": f"Chat {chat_id} deleted"}


@mcp.tool()
async def add_message(
    chat_id: int,
    content: str,
    role: Literal["user", "assistant", "system"] = "user",
    ctx: Context = None
) -> dict[str, Any]:
    """
    Append a message to a chat, updating chat recency.
    
    Args:
        chat_id: The chat ID to add the message to
        content: Message content (required, min 1 char)
        role: Message role (user, assistant, or system; default: user)
    
    Returns:
        Dictionary with message id, chat_id, role, content, and timestamp
    """
    with get_db() as db:
        chat = db.get(Chat, chat_id)
        if not chat:
            return {"error": "Chat not found", "chat_id": chat_id}
        
        try:
            message = Message(
                chat_id=chat_id,
                role=role,
                content=content
            )
            chat.updated_at = utcnow()
            db.add_all([chat, message])
            db.commit()
            db.refresh(message)
            await ctx.log(f"Added message `{message.id}` to chat `{chat_id}`")
            return to_dict(message)
        except ValueError as e:
            return {"error": str(e)}


@mcp.tool()
async def create_feedback(
    chat_id: int,
    rating: int,
    sentiment: str,
    comment: str,
    ctx: Context = None
) -> dict[str, Any]:
    """
    Attach feedback to a chat (one per chat). Update feedback if it already exists.
    
    Args:
        chat_id: The chat ID to attach feedback to
        rating: Rating from 1 to 5
        sentiment: Sentiment (positive, neutral, or negative)
        comment: Comment (max 4000 chars)
    
    Returns:
        Dictionary with feedback data or error message
    """
    with get_db() as db:
        chat = db.get(Chat, chat_id)
        if not chat:
            return {"error": "Chat not found", "chat_id": chat_id}
        
        try:
            feedback = db.query(Feedback).filter(Feedback.chat_id == chat_id).first()
            if feedback:
                feedback.rating, feedback.sentiment, feedback.comment = rating, sentiment, comment
            else:
                feedback = Feedback(chat_id=chat_id, rating=rating, sentiment=sentiment, comment=comment)
                db.add(feedback)
            chat.updated_at = utcnow()
            db.commit()
            db.refresh(feedback)
            await ctx.log(f"Added/Updated feedback for chat `{chat_id}`")
            return to_dict(feedback)
        except ValueError as e:
            return {"error": str(e)}
        
@mcp.tool()
async def export_chat_history(
    user_id: str,
    ctx: Context = None
) -> dict[str, Any]:
    """
    Export a chat's complete history to file with progress tracking.
    
    Args:
        user_id: The user ID whose chat history to export
    
    Returns:
        Dictionary with successful status and file path of the exported chat history
    """
    import json
    _format = "json"
    # Start progress
    await ctx.report_progress(
        progress=0,
        total=100,
        message="Starting export..."
    )
    await asyncio.sleep(0.5)  # simulate delay

    with get_db() as db:
        chats = db.query(Chat).filter(Chat.user_id == user_id).order_by(Chat.updated_at.desc()).all()  
        num_chats = len(chats)      
        result = []
        for chat in chats:
            chat_data = to_dict(chat)        
            # feedback    
            feedback = db.query(Feedback).filter(Feedback.chat_id == chat.id).first()
            chat_data["feedback"] = to_dict(feedback)
            # msg
            messages = (
                db.query(Message)
                .filter(Message.chat_id == chat.id)
                .order_by(Message.created_at.asc())
                .all()
            )
            chat_data["messages"] = [to_dict(msg) for msg in messages]            
            result.append(chat_data)  
            await ctx.report_progress(
                progress=int((len(result)/num_chats)*80),
                total=100,
                message=f"Exported {len(result)}/{num_chats} chats..."
            )
            await asyncio.sleep(0.05)  # simulate delay
        
        # Format data
        await asyncio.sleep(0.5)  # simulate delay
        await ctx.report_progress(
            progress=90,
            total=100,
            message=f"Formatting as {_format}..."
        )
        
        export_dir = os.path.join(_base_dir, "tmp", "exports")
        os.makedirs(export_dir, exist_ok=True)
        filename = f"chat_{user_id}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.{_format}"
        filepath = os.path.join(export_dir, filename)
        
        # Write file
        await asyncio.sleep(0.5)  # simulate delay
        await ctx.report_progress(
            progress=95,
            total=100,
            message="Writing file..."
        )        

        with open(filepath, "w") as f:
            f.write(json.dumps(result, indent=2, default=str))
        
        # Complete
        await asyncio.sleep(0.5)  # simulate delay
        await ctx.report_progress(
            progress=100,
            total=100,
            message="Export complete!"
        )
        
        await ctx.log(f"Exported chat for user `{user_id}` to {filepath}")
        
        return {
            "success": True,
            "filepath": filepath
        }

# select transport
if args.transport and args.transport.lower() == "stdio":
    mcp.run()
else:
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    import uvicorn
    # add cors for js clients, like MCP inspector
    middlewares = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["mcp-session-id"],
        )
    ]
    app = mcp.http_app(middleware=middlewares)
    uvicorn.run(app, host="0.0.0.0", port=8000,reload=False)
    
    

