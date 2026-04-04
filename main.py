# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:41:31 2024

@author: USER
"""

import os

from app import app, logger, handle_exceptions


task_id = 1

Stage_Dict = {
    1: "",
    2: ""
}

@handle_exceptions
def run(stage_id, stage_name):
    match stage_id:
        case 1:
            logger.info(f'>>>>> {stage_name} started <<<<<')
            logger.info(f'>>>>> {stage_name} completed <<<<<')
        
        case 2:
            logger.info(f'>>>>> {stage_name} started <<<<<')
            logger.info(f'>>>>> {stage_name} completed <<<<<')


if __name__ == '__main__':
    task_ids = list(Stage_Dict.keys())
    if task_id in task_ids:
        task_name = Stage_Dict[task_id]
        if isinstance(task_name, str):
            run(task_id, task_name)
        else:
            for n in task_name.keys():
                stage_name = task_name[n]
                run(n, stage_name)

