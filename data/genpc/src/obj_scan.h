/*
 * obj_scan.h
 *
 *  Created on: 2020年2月18日
 *      Author: wuhang
 */

#ifndef OBJ_SCAN_H_
#define OBJ_SCAN_H_

#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkImageActor.h>
#include <vtkLight.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkWindowToImageFilter.h>
#include <vtkInteractorObserver.h>
#include <vtkPLYReader.h>
#include <vtkOBJReader.h>
#include <vtkPolyDataReader.h>
#include <vtkInteractorObserver.h>
#include <vtkCellLocator.h>
#include <vtkOBBTree.h>
#include <vtkLine.h>
#include <vtkWorldPointPicker.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkGeneralTransform.h>
#include <vtkPolyDataWriter.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/filters/voxel_grid.h>

using namespace std;

class obj_scan{
private:
	bool rendered;
	int pixel[2];
	double viewpoint[3];
	double targetpoint[3];
	vtkPolyData *data;
	vtkSmartPointer<vtkOBJReader> reader = vtkOBJReader::New();
	vtkSmartPointer<vtkActor> cylinderActor=vtkSmartPointer<vtkActor>::New();
	vtkSmartPointer<vtkPolyDataMapper> cylinderMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
	vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();  //管理对象的渲染场景（相机，光照等，世界坐标系
	vtkSmartPointer<vtkRenderWindow> renWin = vtkSmartPointer<vtkRenderWindow>::New();  //将渲染场景连接至操作系统
	vtkSmartPointer<vtkOBBTree> obbtree = vtkSmartPointer<vtkOBBTree>::New();
	vtkSmartPointer<vtkCellLocator> cellLocator = vtkSmartPointer<vtkCellLocator>::New();
	vtkSmartPointer<vtkTransform> trans = vtkSmartPointer<vtkTransform>::New();
    vtkSmartPointer<vtkTransformPolyDataFilter> PolyDataFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();


public:
	float theta_y,theta_z;
	obj_scan(string const& filename, int *imgsize, double * cameraPt, double *targetPt);
	void rotate(float theta_1, float theta2);
	void render(double const& ViewAngle);
	void world2view(double *intersection, double *displayPt);
	void OBBTreeintersect(double *intersection, double *tpt, vector<vector<double>>& scanpc);
	void OBBTreeintersect(double *intersection, double *viewpoint2, double *tpt, vector<vector<double>>& scanpc);
	void CellLocatorintersect(double *intersection, double *tpt, vector<vector<double>>& scanpc);
	void intersect(double *intersection, double *tpt, vector<vector<double>>& scanpc, string mode);
	void vtk2cv2(string const& savename);
	void cvShow (string name, cv::Mat & img);
	void showdepth(cv::Mat & renderedImg, double *displayPt);
	void showdepth(cv::Mat & renderedImg, vector<vector<double>> const & depthmat);
	~obj_scan(){cout<<"Class deleting... "<<endl;};
};



#endif /* OBJ_SCAN_H_ */
