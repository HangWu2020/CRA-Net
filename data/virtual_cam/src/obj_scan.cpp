/*
 * obj_scan.cpp
 *
 *  Created on: 2020年2月18日
 *      Author: wuhang
 */


# include "obj_scan.h"

using namespace std;


obj_scan::obj_scan(string const& filename){
	reader->SetFileName(filename.c_str());
	reader->Update();
	data = reader->GetOutput();
	cout << "Load data from " << filename << endl;
	cout << "Number of ply points: " << data->GetNumberOfPoints() << endl;
}



void obj_scan::OBBTreeintersect(double *intersection, double *tpt, vector<vector<double>>& scanpc){
	vtkSmartPointer<vtkPoints> intersectPoints = vtkSmartPointer<vtkPoints>::New();
	// Create the locator
	static bool treeflag = false;
	if (treeflag==false){
		obbtree->SetDataSet(data);
		obbtree->BuildLocator();
		treeflag = true;
		cout << "Setting OBBtree for object" << endl;
	}
	obbtree->IntersectWithLine(viewpoint, tpt, intersectPoints, NULL);
	static int pointid = 0;
	vector<double> scanpt;
	scanpt.clear();
	if (intersectPoints->GetNumberOfPoints()>0){
		pointid += 1;
		intersectPoints->GetPoint(0, intersection);
		scanpt.push_back(intersection[0]);
		scanpt.push_back(intersection[1]);
		scanpt.push_back(intersection[2]);
		scanpc.push_back(scanpt);
	}
}

void obj_scan::CellLocatorintersect(double *intersection, double *tpt, vector<vector<double>>& scanpc){
	static bool locatorflag = false;
	if (locatorflag == false){
		cellLocator->SetDataSet(data);
		cellLocator->BuildLocator();
		locatorflag = true;
	}
	double p_coords[3], t;
	int subId;
	vtkIdType cellId;
	vector<double> scanpt;
	scanpt.clear();
	if (cellLocator->IntersectWithLine(viewpoint, tpt, 0, t, intersection, p_coords, subId, cellId)>0){
		scanpt.push_back(intersection[0]);
		scanpt.push_back(intersection[1]);
		scanpt.push_back(intersection[2]);
		scanpc.push_back(scanpt);
	}
}


void obj_scan::intersect(double *intersection, double *cameraPt, double *tpt, vector<vector<double>>& scanpc, string mode = "OBBTree"){
	viewpoint[0] = cameraPt[0];
	viewpoint[1] = cameraPt[1];
	viewpoint[2] = cameraPt[2];
	if (mode == "OBBTree"){
		obj_scan::OBBTreeintersect(intersection, tpt, scanpc);
	}
	else if (mode == "CellLocator"){
		obj_scan::CellLocatorintersect(intersection, tpt, scanpc);
	}
}

