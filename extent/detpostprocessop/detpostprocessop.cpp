#include <iostream>
#include "defines.h"
#include <cstdio> 
#include <algorithm>
using namespace antgo;

void meshgrid(int height,int width, float* xg, float* yg){
    for(int y_i=0; y_i<height; ++y_i){
        for(int x_i=0; x_i<width; ++x_i){
            xg[y_i*width+x_i] = x_i;
            yg[y_i*width+x_i] = y_i;
        }
    }
}

template<class T>
struct SortElement{
	SortElement(){};
	SortElement(T v,unsigned int i):value(v),index(i){};
	T value;
	unsigned int index;
};

template<typename T>
struct DescendingSort{
	typedef  T			ElementType;
	bool operator()(const SortElement<T>& a,const SortElement<T>& b){
		return a.value > b.value;
	}
};

std::vector<unsigned int> sort(std::vector<std::vector<float>>& data){
    // num*5
	std::vector<SortElement<float>> temp_vector(data.size());
	unsigned int index = 0;
	for (unsigned int i = 0; i < data.size(); ++i){
        temp_vector[i] = SortElement<float>(data[i][4], i);
	}

	//sort
	DescendingSort<float> compare_op;
	std::sort(temp_vector.begin(),temp_vector.end(),compare_op);

	std::vector<unsigned int> result_index(data.size());
	index = 0;
	typename std::vector<SortElement<float>>::iterator iter,iend(temp_vector.end());
	for (iter = temp_vector.begin(); iter != iend; ++iter){
		result_index[index] = ((*iter).index);
		index++;
	}

	return result_index;
}

std::vector<float> get_ious(std::vector<std::vector<float>>& all_bbox, std::vector<float>& target_bbox, std::vector<unsigned int> order, unsigned int offset){
    std::vector<float> iou_list;
    for(unsigned int i=offset; i<order.size(); ++i){
        int index = order[i];
        float inter_x1 = std::max(all_bbox[index][0], target_bbox[0]);
        float inter_y1 = std::max(all_bbox[index][1], target_bbox[1]);

        float inter_x2 = std::min(all_bbox[index][2], target_bbox[2]);
        float inter_y2 = std::min(all_bbox[index][3], target_bbox[3]);

        float inter_w = std::max(inter_x2 - inter_x1, 0.0f);
        float inter_h = std::max(inter_y2 - inter_y1, 0.0f);

        float inter_area = inter_w*inter_h;
        float a_area = (all_bbox[index][2] - all_bbox[index][0])*(all_bbox[index][3] - all_bbox[index][1]);
        float b_area = (target_bbox[2] - target_bbox[0])*(target_bbox[3] - target_bbox[1]);
        float iou = inter_area / (a_area+b_area-inter_area);
        iou_list.push_back(iou);
    }

    return iou_list;
}

std::vector<unsigned int> nms(std::vector<std::vector<float>>& dets, float thresh) {
	std::vector<unsigned int> order = sort(dets);
	std::vector<unsigned int> keep;

	while (order.size() > 0) {
		unsigned int index = order[0];
		keep.push_back(index);
		if (order.size() == 1) {
			break;
		}

        std::vector<float> check_ious = get_ious(dets, dets[index], order, 1);
        std::vector<unsigned int> remained_order;
        for(int i=0; i<check_ious.size(); ++i){
            if(check_ious[i] < thresh){
                remained_order.push_back(order[i + 1]);
            }
        }
		order = remained_order;
	}

	return keep;
}


ANTGO_FUNC void detpostprocess_func(const CITensor* model_size, const CUCTensor* image, const CFTensor* data, const CITensor* level_hw, const CITensor* level_strides, CFTensor* bboxes, CITensor* labels){
    // poss
    int level_num = 3;
    int offset = 0;
    float x_scale = image->dims[1] / (float)(model_size->data[1]);
    float y_scale = image->dims[0] / (float)(model_size->data[0]);

    // data shape, 1xNx7
    float* temp_data = new float[data->dims[0]*data->dims[1]*data->dims[2]];
    std::cout<<"data->dims[0]*data->dims[1]*data->dims[2] "<<data->dims[0]<<" "<<data->dims[1]<<" "<<data->dims[2]<<std::endl;
    memcpy(temp_data, data->data, sizeof(float)*data->dims[0]*data->dims[1]*data->dims[2]);
    for(int level_i=0; level_i<level_num; ++level_i){
        int h = level_hw->data[level_i*2+0];
        int w = level_hw->data[level_i*2+1];
        int stride = level_strides->data[level_i];

        float* xg = new float[h*w];
        float* yg = new float[h*w];
        meshgrid(h,w,xg,yg);

        for(int start_i=offset; start_i<offset+h*w; ++start_i){
            temp_data[start_i*7+0] = (data->data[start_i*7+0] + xg[start_i-offset]) * stride;
            temp_data[start_i*7+1] = (data->data[start_i*7+1] + yg[start_i-offset]) * stride;

            temp_data[start_i*7+2] = exp(data->data[start_i*7+2]) * stride;
            temp_data[start_i*7+3] = exp(data->data[start_i*7+3]) * stride;
        }

        delete[] xg;
        delete[] yg;
        offset += h*w;
    }

    // nms
    int num = data->dims[1];
    std::vector<std::vector<float>> person_bboxes;
    std::vector<std::vector<float>> ball_bboxes;
    for(int i=0; i<num; ++i){
        float* ptr = temp_data + i*7;
        float cx = ptr[0];
        float cy = ptr[1];
        float w = ptr[2];
        float h = ptr[3];
        float obj_pred = ptr[4];
        float cls_0_pred = ptr[5];
        float cls_1_pred = ptr[6];

        // obj_pred 0.2 best
        if (obj_pred > 0.1){
            if(cls_0_pred > cls_1_pred && cls_0_pred > 0.1){
                person_bboxes.push_back({cx-w/2,cy-h/2,cx+w/2,cy+h/2, cls_0_pred});
            }
            else if(cls_0_pred < cls_1_pred && cls_1_pred > 0.3){
                ball_bboxes.push_back({cx-w/2,cy-h/2,cx+w/2,cy+h/2, cls_1_pred});
            }
        }
    }

    if(person_bboxes.size() > 0){
        // preson 
        std::vector<unsigned int> filter_person_index;
        filter_person_index = nms(person_bboxes, 0.4);
        std::vector<std::vector<float>> filter_person_bboxes;
        for(int i=0; i<filter_person_index.size(); ++i){
            filter_person_bboxes.push_back(person_bboxes[filter_person_index[i]]);
        }
        person_bboxes = filter_person_bboxes;
    }
    if(ball_bboxes.size() > 0){
        // ball
        std::vector<unsigned int> filter_ball_index;
        filter_ball_index = nms(ball_bboxes, 0.01);
        std::vector<std::vector<float>> filter_ball_bboxes;
        for(int i=0; i<filter_ball_index.size(); ++i){
            filter_ball_bboxes.push_back(ball_bboxes[filter_ball_index[i]]);
        }
        ball_bboxes = filter_ball_bboxes;        
    }
    // 合并结果
    int person_num = person_bboxes.size();
    int ball_num = ball_bboxes.size();
    int person_and_ball_num = person_num+ball_num;

    bboxes->create2d(person_and_ball_num, 5);
    labels->create1d(person_and_ball_num);
    for(int i=0; i<person_and_ball_num; ++i){
        if(i<person_bboxes.size()){
            bboxes->data[i*5+0] = person_bboxes[i][0] * x_scale;
            bboxes->data[i*5+1] = person_bboxes[i][1] * y_scale;
            bboxes->data[i*5+2] = person_bboxes[i][2] * x_scale;
            bboxes->data[i*5+3] = person_bboxes[i][3] * y_scale;
            bboxes->data[i*5+4] = person_bboxes[i][4];

            labels->data[i] = 0;
        }
        else{
            bboxes->data[i*5+0] = ball_bboxes[i-person_num][0] * x_scale;
            bboxes->data[i*5+1] = ball_bboxes[i-person_num][1] * y_scale;
            bboxes->data[i*5+2] = ball_bboxes[i-person_num][2] * x_scale;
            bboxes->data[i*5+3] = ball_bboxes[i-person_num][3] * y_scale;
            bboxes->data[i*5+4] = ball_bboxes[i-person_num][4];

            labels->data[i] = 1;
        }
    }

    delete[] temp_data;
}