from typing import List

from ..data_types.cloud import Cloud, LabelledCloud


def collate(clouds: List[Cloud | LabelledCloud]):

  return clouds[0]