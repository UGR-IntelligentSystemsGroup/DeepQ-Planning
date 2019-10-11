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
    MILLIS_IN_MIN = 60*1000
    START_TIME = 1000000
    INITIALIZATION_TIME = 1000000
    ACTION_TIME = 100*60*MILLIS_IN_MIN # No hay límite de tiempo para devolver una acción
    ACTION_TIME_DISQ = 100*60*MILLIS_IN_MIN

    # TOTAL_LEARNING_TIME = 100*60*MILLIS_IN_MIN # 100 horas de entrenamiento (como máximo)  # Uncomment this if the model has yet to be trained
    TOTAL_LEARNING_TIME = 0.1*MILLIS_IN_MIN # Uncomment this if the model is already trained -> skips training phase (except for the first level)
    
    EXTRA_LEARNING_TIME = 1000000
    SOCKET_PORT = 8080
    SCREENSHOT_FILENAME = "gameStateByBytes.png"
