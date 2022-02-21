import config


def check_float(potential_float):
    try:
        float(potential_float)
        return True

    except ValueError:
        return False


def x_transform(x_array):
    """
        This function transforms a numpy array x-coordinates from ARK to ARGoS coordinate system
    """
    x_trans = ((x_array - (config.ARENA_SIZE / 2)) / config.M_TO_PIXEL) / config.CM_TO_M
    return x_trans


def y_transform(y_array):
    """
        This function transforms a numpy array y-coordinates from ARK to ARGoS coordinate system
    """
    y_trans = (((config.ARENA_SIZE / 2) - y_array) / config.M_TO_PIXEL) / config.CM_TO_M
    return y_trans
