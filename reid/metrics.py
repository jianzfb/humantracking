from antgo.framework.helper.runner.builder import *
import numpy as np
import torch


@MEASURES.register_module()
class AccuracyEval(object):
    def __init__(self, topk=(1, ), thrs=0.) -> None:
        self.topk = topk
        self.thrs = thrs

    def keys(self):
        # 约束使用此评估方法，需要具体的关键字信息
        return {'pred': ['pred'], 'gt': ['label']}

    def __call__(self, preds, gts):
        preds_reformat = [pred['pred'] for pred in preds]
        gts_reformat = [gt['label'] for gt in gts]
        preds = np.stack(preds_reformat, 0)
        gts = np.array(gts_reformat).astype(np.int32)

        maxk = max(self.topk)
        num = preds.shape[0]

        static_inds = np.indices((num, maxk))[0]
        pred_label = preds.argpartition(-maxk, axis=1)[:, -maxk:]
        pred_score = preds[static_inds, pred_label]

        sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
        pred_label = pred_label[static_inds, sort_inds]
        pred_score = pred_score[static_inds, sort_inds]

        eval_values = {}
        for k in self.topk:
            correct_k = pred_label[:, :k] == gts.reshape(-1, 1)
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > self.thrs)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            eval_values[f'top_{k}'] = _correct_k.sum() * 100. / num

        return eval_values


#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


@MEASURES.register_module()
class ReidEval(object):
    def keys(self):
        return {'pred': ['feature'], 'gt': ['tag', 'camera', 'label']}

    def __call__(self, preds, gts):
        query_feature = []
        gallery_feature = []
        query_cam = []
        gallery_cam = []
        query_label = []
        gallery_label = []

        for pred, gt in zip(preds, gts):
            if gt['tag'] == 0:
                #  query
                query_feature.append(pred['feature'])
                query_cam.append(int(gt['camera']))
                query_label.append(int(gt['label']))
            else:
                # gallery
                gallery_feature.append(pred['feature'])
                gallery_cam.append(int(gt['camera']))
                gallery_label.append(int(gt['label']))

        query_feature = torch.FloatTensor(np.stack(query_feature, 0))
        gallery_feature = torch.FloatTensor(np.stack(gallery_feature, 0))

        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()

        query_cam = np.array(query_cam)
        query_label = np.array(query_label)
        gallery_cam = np.array(gallery_cam)
        gallery_label = np.array(gallery_label)

        print(query_feature.shape)
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        #print(query_label)
        for i in range(len(query_label)):
            ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
            if CMC_tmp[0]==-1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            #print(i, CMC_tmp[0])

        CMC = CMC.float()
        CMC = CMC/len(query_label) #average CMC
        print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
        return {
            'Rank@1': float(CMC[0].float()),
            'Rank@5': float(CMC[4].float()),
            'Rank@10': float(CMC[9].float()),
            'mAP': float(ap/len(query_label))
        }