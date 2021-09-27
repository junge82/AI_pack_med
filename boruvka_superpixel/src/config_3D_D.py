from config import Config, Feature


class Config_3D_D(Config):

    
    ROOT_FOLDER = '/home/fothar/DAVIS/JPEGImages/480p/'
    OUT_ROOT_FOLDER = '/home/fothar/Boruvka3D/'
    SUPERPIXELS = [100, 200, 500, 1000]

    ROOT_OF_FOLDER = '/home/fothar/pwc_out/DAVIS/'
    OF_EDGE_PREFACTORS = [0.125]

    ROOT_OF_REVERSE_FOLDER = '/home/fothar/pwc_r_out/DAVIS/'

    OF_TOLERANCE_SQES = [1, 4]
    OF_REL_TOLERANCES = [0.10, 0.20]


    FEATURES = [Feature("OF", ROOT_OF_FOLDER, [0.125, 0.25])]

