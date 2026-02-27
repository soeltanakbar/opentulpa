"""API request/response schema models."""

from opentulpa.api.schemas.approvals import (
    ApprovalDecideRequest,
    ApprovalEvaluateRequest,
    ApprovalExecuteRequest,
    ApprovalPendingStatusQuery,
)
from opentulpa.api.schemas.chat import InternalChatRequest
from opentulpa.api.schemas.files import (
    FileAnalyzeRequest,
    FileGetRequest,
    FileSearchRequest,
    FileSendLocalRequest,
    FileSendRequest,
    FileSendWebImageRequest,
)
from opentulpa.api.schemas.memory import MemoryAddRequest, MemorySearchRequest
from opentulpa.api.schemas.profiles import (
    DirectiveClearRequest,
    DirectiveGetRequest,
    DirectiveSetRequest,
    TimeProfileGetRequest,
    TimeProfileSetRequest,
)
from opentulpa.api.schemas.scheduler import (
    RoutineCreateRequest,
    RoutineDeleteWithAssetsRequest,
    SchedulerRoutineDeleteQuery,
    SchedulerRoutinesQuery,
)
from opentulpa.api.schemas.skills import (
    SkillDeleteRequest,
    SkillGetRequest,
    SkillListRequest,
    SkillUpsertRequest,
)
from opentulpa.api.schemas.tasks import TaskCreateRequest, TaskEventsQuery, TaskRelaunchRequest
from opentulpa.api.schemas.telegram import TelegramWebhookRequest
from opentulpa.api.schemas.tulpa import (
    TulpaReadFileQuery,
    TulpaRunTerminalRequest,
    TulpaValidateFileRequest,
    TulpaWriteFileRequest,
)
from opentulpa.api.schemas.wake_search import WakePayload, WebSearchRequest

__all__ = [
    "ApprovalDecideRequest",
    "ApprovalEvaluateRequest",
    "ApprovalExecuteRequest",
    "ApprovalPendingStatusQuery",
    "DirectiveClearRequest",
    "DirectiveGetRequest",
    "DirectiveSetRequest",
    "FileAnalyzeRequest",
    "FileGetRequest",
    "FileSearchRequest",
    "FileSendLocalRequest",
    "FileSendRequest",
    "FileSendWebImageRequest",
    "InternalChatRequest",
    "MemoryAddRequest",
    "MemorySearchRequest",
    "RoutineCreateRequest",
    "RoutineDeleteWithAssetsRequest",
    "SchedulerRoutineDeleteQuery",
    "SchedulerRoutinesQuery",
    "SkillDeleteRequest",
    "SkillGetRequest",
    "SkillListRequest",
    "SkillUpsertRequest",
    "TaskCreateRequest",
    "TaskEventsQuery",
    "TaskRelaunchRequest",
    "TelegramWebhookRequest",
    "TimeProfileGetRequest",
    "TimeProfileSetRequest",
    "TulpaReadFileQuery",
    "TulpaRunTerminalRequest",
    "TulpaValidateFileRequest",
    "TulpaWriteFileRequest",
    "WakePayload",
    "WebSearchRequest",
]
