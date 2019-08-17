import sys


class CompetitionParameters:
    """
     * Competition parameters, should be the same with the ones on Server side:
     * refer to core.competition.CompetitionParameters
    """
    def __init__(self):
        pass

    if 'win32' in sys.platform:
        OS_WIN = True
    else:
        OS_WIN = False

    USE_SOCKETS = True
    START_TIME = 1000000
    INITIALIZATION_TIME = 1000000
    ACTION_TIME = 500000
    ACTION_TIME_DISQ = 510000
    MILLIS_IN_MIN = 60*1000
    # TOTAL_LEARNING_TIME = 180*MILLIS_IN_MIN # 3 horas de entrenamiento
    TOTAL_LEARNING_TIME = 0.5*MILLIS_IN_MIN # Casi nada de entrenamiento -> solo quiero probar cómo funciona el model en el validation set
    EXTRA_LEARNING_TIME = 1000
    SOCKET_PORT = 8080
    SCREENSHOT_FILENAME = "gameStateByBytes.png"
