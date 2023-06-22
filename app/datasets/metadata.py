from typing import Optional, Dict, Union, Type, Tuple, Callable


class Metadata(dict):
    @staticmethod
    def collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        return batch
