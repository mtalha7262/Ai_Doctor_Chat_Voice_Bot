from opcode import opname
import os
from pydub import AudioSegment
from gtts import gTTS
from setup_multimodel_file import analyze_image_with_query
import elevenlabs
from elevenlabs.client import ElevenLabs
import subprocess
import platform
