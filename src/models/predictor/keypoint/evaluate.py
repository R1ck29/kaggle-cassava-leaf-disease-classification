import os
import numpy as np
from munkres import Munkres
from torch.utils.data import Dataset

class Evaluator(object):
    def __init__(self, num_joints, conf_thresh=0.2, head_neck=None,
                 static_thresh=100, h_ratio=0.5, oks_ratio=0.1, oks_thresh=0.5):
        """keypoint評価用のクラス。

        Args:
            conf_thresh (float): 予測結果のconfidence threshold。
            head_neck    (tuple): headとneckのID、PCK算出時にheadの大きさを使う場合(PCKh)
            static_thresh  (int): PCK算出時に固定のthreshを使う場合の値。
            h_ratio      (float): PCK算出時のfactor
            oks_ratio    (float): OKS算出時のfactor
            oks_thresh   (float): OKSを用いた評価時の閾値。0.5であればAP@OKS=0.5が算出できる。
            
        """
        super().__init__()
        self.num_joints = num_joints
        self.conf_thresh = conf_thresh
        #PCKhで計算したい場合は、head_idとneck_idを渡す
        self.head_neck = head_neck
        self.h_ratio = h_ratio
        self.static_thresh = static_thresh
        self.oks_ratio = oks_ratio
        self.oks_thresh = oks_thresh
        #ハンガリアン法によるGTと予測のマッチング
        self.matcher = Munkres()
        
    def calc_dist(self, preds, gts):
        """予測とGTのユークリッド距離を算出する.

        Args:
            preds (array): 予測結果。(obj数, kpt数, 3[y座標、x座標、conf])、3次元の配列。
            gts   (array): GT。(obj数, kpt数, 3[y座標、x座標、visibility])、3次元の配列。
            
        Returns:
            dist_mat (array): 予測結果とGTのユークリッド距離. (GTのobj数、予測のobj数、kpt数)、3次元の配列。
        """
        dist = np.array([preds[...,:2]-gt for gt in gts[...,:2]])
        dist_mat = np.linalg.norm(dist, axis=3)
        
        return dist_mat
        
    def calc_oks(self, preds, gts):
        """予測とGTのOKS(Object Keypoint Similarity)を算出する.

        Args:
            preds (array): 予測結果。(obj数, kpt数, 3[y座標、x座標、conf])、3次元の配列。
            gts   (array): GT。(obj数, kpt数, 3[y座標、x座標、visibility])、3次元の配列。
            
        Returns:
            mat        (array): 予測結果とGTのOKS。(GTのobj数、予測のobj数)、2次元の配列。
            thresh_list (list): GT毎に設定される距離の閾値。対象objの対角距離をベースに算出。
            dist_mat   (array): 予測結果とGTのユークリッド距離. (GTのobj数、予測のobj数、kpt数)、3次元の配列。
        """
        mat = []
        thresh_list = []
        
        dist_mat = self.calc_dist(preds, gts)
        
        for gid, gt in enumerate(gts):
            dist = dist_mat[gid]

            maximum = np.max(gt[...,:2], axis=0)
            minimum = np.min(gt[...,:2], axis=0)

            diagonal = np.linalg.norm(maximum-minimum)
            thresh = diagonal * self.oks_ratio
            
            okss = np.exp((- dist**2)/(2*thresh**2))
            
            #conf_thresh以下の予測は採用しない
            okss[preds[...,2] < self.conf_thresh] = 0
            
            #GTのラベルが0(invisible)なら結果にカウントしない
            del_id = np.where(gt[...,2]==0)[0]
            okss = np.delete(okss, obj=del_id, axis=1)
            
            mat.append(np.mean(okss, axis=1))
            thresh_list.append(thresh)
            
        return np.array(mat), thresh_list, dist_mat
    
    def calc_pck(self, preds, gts):
        """予測とGTのPCK(Percentage of Correct Keypoints)を算出する.

        Args:
            preds (array): 予測結果。(obj数, kpt数, 3[y座標、x座標、conf])、3次元の配列。
            gts   (array): GT。(obj数, kpt数, 3[y座標、x座標、visibility])、3次元の配列。
            
        Returns:
            mat        (array): 予測結果とGTのPCK。(GTのobj数、予測のobj数)、2次元の配列。
            thresh_list (list): GT毎に設定される距離の閾値。対象objの頭の大きさをベースに算出、あるいは固定値(static_thresh)が入る。
            dist_mat   (array): 予測結果とGTのユークリッド距離. (GTのobj数、予測のobj数、kpt数)、3次元の配列。
        """
        mat = []
        thresh_list = []
        
        dist_mat = self.calc_dist(preds, gts)
        
        for gid, gt in enumerate(gts):
            thresh = 0
            dist = dist_mat[gid]
            
            if self.head_neck:
                thresh = np.linalg.norm(gt[self.head_neck[0],:2] - gt[self.head_neck[1],:2]) * self.h_ratio

            if thresh == 0:
                thresh = self.static_thresh
                
            #conf_thresh以下の予測は採用しない
            #GTのラベルが0(invisible)なら結果にカウントしない
            del_id = np.where(gt[...,2]==0)[0]
            cks = (dist < thresh) & (preds[...,2] > self.conf_thresh)
            cks = np.delete(cks, obj=del_id, axis=1)
            
            mat.append(np.mean(cks, axis=1))
            thresh_list.append(thresh)
        
        return np.array(mat), thresh_list, dist_mat
        
    def calc_match(self, mat):
        """OKSもしくはPCKに基づいて、ハンガリアン法による予測とGTのマッチングを行う。

        Args:
            mat (array):OKSもしくはPCKのスコア。(obj数, kpt数)、2次元の配列。
            
        Returns:
            match (list): マッチした予測結果とGTのIDのペア。
        """
        #GTが予測結果より大きい場合は、paddingする
        diff = mat.shape[0] - mat.shape[1]
        if diff > 0:
            new_mat = np.hstack([mat, np.zeros((mat.shape[0], diff))])
            match = self.matcher.compute(1 - new_mat)
            #paddingによって生成されたidの削除
            ind = [(pid not in range(mat.shape[1], mat.shape[0])) for pid in np.array(match)[:,1]]
            match = np.array(match)[ind]

        else:
            match = np.array(self.matcher.compute(1 - mat))
            
        return match
    
    def calc_confmat(self, preds, gts, metrics='oks'):
        """confusion matrix[TP, FP, FN, TN]の計算。

        Args:
            preds (array): 予測結果。(obj数, kpt数, 3[y座標、x座標、conf])、3次元の配列。
            gts   (array): GT。(obj数, kpt数, 3[y座標、x座標、visibility])、3次元の配列。
            
        Returns:
            confmat (array): confusin matrix。metircsにOKSを選択した場合と、PCKを選択した場合でreturnが異なる。
                             OKSを選択した場合は、(tp, fp, fn, tn)、1次元の配列。
                             PCKを選択した場合は、(kpt数、4[tp, fp, fn, tn])、2次元の配列。
        """
        #predsが空arrayの場合、評価用にダミーの予測結果を用意する
        if len(preds)==0:
            preds = np.zeros((1, self.num_joints, 3))
        
        pid = np.arange(len(preds))
        gid = np.arange(len(gts))
        
        if metrics=='pck':
            #kpt数 * 4[TP, FP, FN, TN]
            mat, thresh_list, dist_mat =  self.calc_pck(preds, gts)
            match = self.calc_match(mat)
            out_pid = list(set(pid) - set(match[:,1]))
            out_gid = list(set(gid) - set(match[:,0]))

            confmat = np.zeros((self.num_joints, 4))
            for gid, pid in match:
                #diffの算出
                diff = dist_mat[gid, pid]
                conf = preds[pid,:,2]
                visbility = gts[gid,:,2]
                thresh = thresh_list[gid]
                #TP(距離が閾値内、かつconfidenceが閾値以上)　かつ gtがvisible
                confmat[:,0] += ((diff<=thresh)&(conf>=self.conf_thresh)&(visbility!=0)).astype(int)
                #FP(距離が閾値外、かつconfidenceが閾値以上)　かつ　gtがvisible
                confmat[:,1] += ((diff>thresh)&(conf>=self.conf_thresh)&(visbility!=0)).astype(int)
                #FN(予測が外れている、あるいは予測されていない)　かつ　gtがvisible
                confmat[:,2] += ((diff>thresh)&(conf>=self.conf_thresh)&(visbility!=0)).astype(int) +\
                                ((conf<self.conf_thresh)&(visbility!=0)).astype(int)

            for pid in out_pid:
                #もしconf_thresh以上ならFP
                conf = preds[pid,:,2]
                confmat[:,1] += (conf>=self.conf_thresh).astype(int)

            for gid in out_gid:
                #FN
                confmat[:,2] += np.ones(cfg.MODEL.NUM_JOINTS)
        
        elif metrics=='oks':
            #confmat 4[TP, FP, FN, TN]
            mat, _, _ =  self.calc_oks(preds, gts)
            match = self.calc_match(mat)
            out_pid = list(set(pid) - set(match[:,1]))
            out_gid = list(set(gid) - set(match[:,0]))
            confmat = np.zeros(4)
            for gid, pid in match:
                if mat[gid, pid] > self.oks_thresh:
                    #TP
                    confmat[0] += 1
                else:
                    #FP
                    confmat[1] += 1
                     
            for pid in out_pid:
                #FP
                confmat[1] += 1

            for gid in out_gid:
                #FN
                confmat[2] += 1
        
        return confmat