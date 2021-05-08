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
#include <vtkInteractorObserver.h>
#include <vtkCellLocator.h>
#include <vtkOBBTree.h>
#include <vtkLine.h>
#include <vtkWorldPointPicker.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>

using namespace std;

class obj_scan{
private:
	vtkPolyData *data;
	bool rendered;
	double viewpoint[3];
	vtkSmartPointer<vtkOBJReader> reader = vtkOBJReader::New();
	vtkSmartPointer<vtkActor> cylinderActor=vtkSmartPointer<vtkActor>::New();
	vtkSmartPointer<vtkPolyDataMapper> cylinderMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
	vtkSmartPointer<vtkOBBTree> obbtree = vtkSmartPointer<vtkOBBTree>::New();
	vtkSmartPointer<vtkCellLocator> cellLocator = vtkSmartPointer<vtkCellLocator>::New();


public:
	obj_scan(string const& filename);
	void OBBTreeintersect(double *intersection, double *tpt, vector<vector<double>>& scanpc);
	void CellLocatorintersect(double *intersection, double *tpt, vector<vector<double>>& scanpc);
	void intersect(double *intersection, double *cameraPt, double *tpt, vector<vector<double>>& scanpc, string mode);
	~obj_scan(){cout<<"Class deleting... "<<endl;};
};



#endif /* OBJ_SCAN_H_ */
