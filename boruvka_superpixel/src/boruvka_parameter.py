from os.path import join
import copy
import config

class FeatureParam(object):
    NAME = None

    def __init__(self, name, root_folder, prefactor):
        self.NAME = name
        self.ROOT_FOLDER = root_folder
        self.PREFACTOR = prefactor

    def featurefolder(self, scene_id):
        return join(self.ROOT_FOLDER, scene_id)


class BoruvkaParam(object):
    NAME = None
    
    

    def __init__(self, config, scene_id):
        self.ROOT_FOLDER = config.ROOT_FOLDER
        self.OUT_ROOT_FOLDER = config.OUT_ROOT_FOLDER
        self.SAVE_VIDEO = config.SAVE_VIDEO
        self.config = config
        self.SCENE_ID = scene_id
        self.FEATURES = []
        

    def append_features(self, features, append_list):
        feature = features[0]
        if not append_list:
            append_list = [[f] for f in feature.PREFACTORS]
        else:
            append_list = [l.append(p) for l in append_list for p in feature.PREFACTORS]

        if len(features) > 1:
            return self.append_features(features[1:], append_list)
        else:
            return append_list

    def set_params_base(self, sp_number, prefactors):
        self.SP_NUMBER = sp_number
        for feature, prefactor in zip(self.config.FEATURES, prefactors):
            self.FEATURES.append(FeatureParam(feature.NAME, feature.ROOT_FOLDER, prefactor))
            


    def permutate(self, params):
        append_list = []
        append_list = self.append_features(self.config.FEATURES, append_list)
        for sp_number in self.config.SUPERPIXELS:
            for prefactors in append_list:
                x = copy.deepcopy(self)
                x.set_params_base(sp_number, prefactors)
                params.append(x)

    def infolder(self):
        return join(self.ROOT_FOLDER, self.SCENE_ID)

    def out_folder(self):
        outfolder = join(self.OUT_ROOT_FOLDER, "{}_{}".format(self.SCENE_ID, self.SP_NUMBER))
        for f in self.FEATURES:
            outfolder += "_{}_{}".format(f.NAME, int(f.PREFACTOR * 100))
        return outfolder
     

    
class Boruvka3DParam(BoruvkaParam): 
    NAME = "3D"   
    
    def __init__(self, config, scene_id, params=None):
        super(Boruvka3DParam, self).__init__(config, scene_id)
        if params:
            self.permutate(params)

    def permutate(self, params):
        self.set_params_3d(params)
    
    def set_params_3d(self, params):
        super(Boruvka3DParam, self).permutate(params)

    

    def out_folder(self):
        return super(Boruvka3DParam, self).out_folder() + "_3d"
        
    

class Boruvka3DOFParam(Boruvka3DParam): 
    NAME = "3DOF"   
    
    
    def __init__(self, config, scene_id, params=None):
        super(Boruvka3DOFParam, self).__init__(config, scene_id)
        self.ROOT_OF_FOLDER = self.config.ROOT_OF_FOLDER
        if params:
            self.permutate(params)

    def permutate(self, params):
        for ofedge_prefactor in self.config.OF_EDGE_PREFACTORS:
            x = copy.deepcopy(self)
            x.set_params_of(params, ofedge_prefactor)
    
    def set_params_of(self, params, ofedge_prefactor):
        self.OFEDGE_PREFACTOR = ofedge_prefactor
        super(Boruvka3DOFParam, self).permutate(params)

    def offolder(self):
        return join(self.ROOT_OF_FOLDER, self.SCENE_ID)


    def out_folder(self):
        return super(Boruvka3DOFParam, self).out_folder() + "_of_{}".format(int(self.OFEDGE_PREFACTOR*100))
        
        

class Boruvka3DOFReverseParam(Boruvka3DOFParam):
    NAME = "3DOF_REVERSE"
    

    def __init__(self, config, scene_id, params=None):
        super(Boruvka3DOFReverseParam, self).__init__(config, scene_id)
        self.ROOT_OF_REVERSE_FOLDER = config.ROOT_OF_REVERSE_FOLDER
        if not params==None:
            self.permutate(params)           

    def permutate(self, params):
        for of_tolerance_sq in self.config.OF_TOLERANCE_SQES:
                for of_rel_tolerance in self.config.OF_REL_TOLERANCES:
                    x = copy.deepcopy(self)
                    x.set_params(params, of_tolerance_sq, of_rel_tolerance)
    
    def set_params(self, params, of_tolerance_sq, of_rel_tolerance):
        self.OF_TOLERANCE_SQ = of_tolerance_sq
        self.OF_REL_TOLERANCE = of_rel_tolerance
        super(Boruvka3DOFReverseParam, self).permutate(params)                            
    
    def ofreversefolder(self):
        return join(self.ROOT_OF_REVERSE_FOLDER, self.SCENE_ID)

    def out_folder(self):
        return super(Boruvka3DOFReverseParam, self).out_folder() + "_r_{}_{}".format(self.OF_TOLERANCE_SQ, int(self.OF_REL_TOLERANCE*100))
        