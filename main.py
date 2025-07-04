import os
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import groq, silero, elevenlabs
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env.local")

# Simple health check server for Railway
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<h1>LiveKit Agent is running!</h1><p>This is a voice assistant agent waiting for LiveKit room connections.</p>')
    
    def log_message(self, format, *args):
        # Suppress HTTP request logs
        pass

def start_health_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    print(f"Health check server running on port {port}")
    server.serve_forever()

@function_tool
async def lookup_weather(
    context: RunContext,
    location: str,
):
    """Used to look up weather information."""
    return {"weather": "sunny", "temperature": 70}

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = Agent(
        instructions="""
            You are a friendly voice assistant built by LiveKit.
            Start every conversation by greeting the user.
            Only use the `lookup_weather` tool if the user specifically asks for weather information.
            Never assume a location or provide weather data without a request.
            """,
        tools=[lookup_weather],
    )
    
    session = AgentSession(
        vad=silero.VAD.load(),
        # Using Groq for STT and LLM, ElevenLabs for TTS
        stt=groq.STT(),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=elevenlabs.TTS(
            voice_id="EXAVITQu4vr4xnSDxMaL",  # Bella voice ID
            model="eleven_turbo_v2_5"
        ),
    )

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="Say hello, then ask the user how their day is going and how you can help.")

if __name__ == "__main__":
    # Start health check server in background thread
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    
    # Start LiveKit agent
    worker_options = WorkerOptions(entrypoint_fnc=entrypoint)
    cli.run_app(worker_options)