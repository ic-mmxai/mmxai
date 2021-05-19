class APSchedulerJobConfig(object):
    SCHEDULER_API_ENABLED = True
    JOBS = [
        {
            "id": "No1",
            "func": "file_manage:check_database",
            "args": ("./static/user",),
            "trigger": {
                "type": "interval",
                # "day_of_week": "0-6",
                # "hour": "*",
                # "minute": "1",
                "seconds": 300,
            },
        }
    ]
