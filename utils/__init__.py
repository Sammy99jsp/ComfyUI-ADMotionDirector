from dataclasses import dataclass
from .scheduling import InputSchedule, InputScheduler, INPUT_SCHEDULERS
from .inputs import LabeledVid, TrainingVideo

@dataclass
class VideoPipe:
    """
    Multiple labeled videos, together.
    """
    videos: list[LabeledVid]
    input_scheduler: type[scheduling.InputScheduler]

    @classmethod
    def empty(cls, sched: type[scheduling.InputScheduler]):
        return cls([], sched)
    
    def __len__(self):
        return len(self.videos)
    
    def append(self, vid: LabeledVid):
        self.videos.append(vid)
        return self