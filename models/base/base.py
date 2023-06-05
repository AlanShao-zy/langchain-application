from abc import ABC, abstractmethod
from typing import Optional, List
import traceback
from collections import deque
from queue import Queue
from threading import Thread
import torch
import transformers

from models.loader import LoaderCheckPoint


class AnswerResult:
    """
    message entity
    """
    history: List[List[str]] = []
    llm_output: Optional[dict] = None


class BaseAnswer(ABC):

    @property
    @abstractmethod
    def _check_point(self) -> LoaderCheckPoint:
        """
        model checkpoint
        """
    
    @property
    @abstractmethod
    def _history_len(self) -> int:
        """
        history length
        """
    
    @abstractmethod
    def set_history_len(self, history_len: int) -> None:
        """
        set history length
        """
    
    def generatorAnswer(self, prompt: str, history: List[List[str]] = None, streaming: bool = False):
        pass