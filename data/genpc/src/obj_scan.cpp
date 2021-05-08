/*
 * obj_scan.cpp
 *
 *  Created on: 2020年2月18日
 *      Author: wuhang
 */


# include "obj_scan.h"

using namespace std;


obj_scan::obj_scan(string const& filename, int *imgsize, double * cameraPt, double *targetPt){
	reader->SetFileName(filename.c_str());
	reader->Update();
	data = reader->GetOutput();
	cout << "Load data from " << filename << endl;
	cout << "Number of model points: " << data->GetNumberOfPoints() << endl;
	rendered = false;
	pixel[0] = imgsize[0];
	pixel[1] = imgsize[1];
	viewpoint[0] = cameraPt[0];
	viewpoint[1] = cameraPt[1];
	viewpoint[2] = cameraPt[2];
	targetpoint[0] = targetPt[0];
	targetpoint[1] = targetPt[1];
	targetpoint[2] = targetPt[2];
	obj_scan::theta_y = 0.0;
	obj_scan::theta_z = 0.0;
}

void obj_scan::rotate(float theta_1, float theta_2){
	theta_y = theta_1;
	theta_z = theta_2;
	trans->RotateWXYZ(theta_1, 0, 1, 0);
	trans->RotateWXYZ(theta_2, 0, 0, 1);
	PolyDataFilter->SetInputData(reader->GetOutput());
	PolyDataFilter->SetTransform(trans);
	PolyDataFilter->Update();
	trans->Identity();
}

void obj_scan::render(double const& ViewAngle){
	cylinderMapper->SetInputConnection(reader->GetOutputPort()); 
	cylinderActor->SetMapper(cylinderMapper);
	cylinderActor->GetProperty()->SetColor(0.0,0.0,1.0);  
	cylinderActor->GetProperty()->SetOpacity(1.0);
	renderer->Clear();

	renderer->AddActor(cylinderActor);

	renderer->SetBackground(1.0,1.0,1.0);

	camera->SetViewAngle(ViewAngle);
	camera->SetPosition(viewpoint[0],viewpoint[1],viewpoint[2]);
	camera->SetFocalPoint(targetpoint[0],targetpoint[1],targetpoint[2]);
	if (viewpoint[0] == 0 && viewpoint[2] == 0){
		camera->SetViewUp(0.0, 0.0, 1.0);
	}
	camera->SetParallelProjection(1);
	renderer->ResetCameraClippingRange();
	renderer->SetActiveCamera(camera);

	renWin->AddRenderer(renderer);
	renWin->SetSize(pixel[0],pixel[1]);
	renWin->SetWindowName("RenderImage");
	renWin->SetOffScreenRendering(1);
	renWin->Render();  //Render the pipeline
	rendered = true;

}

void obj_scan::world2view(double *intersection, double *displayPt){
	if (rendered==false){
		cerr << "Object not rendered yet, please render first!" << endl;
		exit(0);
	}
	else{
		vtkInteractorObserver::ComputeWorldToDisplay(renderer, intersection[0], intersection[1], intersection[2], displayPt);
	}

}

void obj_scan::OBBTreeintersect(double *intersection, double *tpt, vector<vector<double>>& scanpc){
	vtkSmartPointer<vtkPoints> intersectPoints = vtkSmartPointer<vtkPoints>::New();
	// Create the locator
	static bool treeflag = false;
	if (treeflag==false){
		obbtree->SetDataSet(data);
		obbtree->BuildLocator();
		treeflag = true;
		//cout << "Turn tree flag to true" << endl;
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

void obj_scan::OBBTreeintersect(double *intersection, double *viewpoint2, double *tpt, vector<vector<double>>& scanpc){
	vtkSmartPointer<vtkPoints> intersectPoints = vtkSmartPointer<vtkPoints>::New();
	// Create the locator
	obbtree->SetDataSet(PolyDataFilter->GetOutput());
	obbtree->BuildLocator();
	obbtree->IntersectWithLine(viewpoint2, tpt, intersectPoints, NULL);
	static int pointid = 0;
	vector<double> scanpt;
	scanpt.clear();
	if (intersectPoints->GetNumberOfPoints() > 0){
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

void obj_scan::vtk2cv2(string const& savename){
	int dim[3];
	vtkSmartPointer<vtkWindowToImageFilter> wif = vtkSmartPointer<vtkWindowToImageFilter>::New();
	wif->SetInput(renWin);
	wif->Update();
	wif->GetOutput()->GetDimensions(dim);
	cv::Mat renderedImg(dim[1],dim[0],CV_8UC3,wif->GetOutput()->GetScalarPointer());
	cv::cvtColor(renderedImg, renderedImg, CV_BGR2RGB);
	cv::flip(renderedImg, renderedImg, 0);
	cv::imwrite(savename,renderedImg);
}

void obj_scan::cvShow(string name, cv::Mat & img){
	cv::imshow(name, img);
	cv::waitKey();
}

void obj_scan::showdepth(cv::Mat & renderedImg, double *displayPt){
	cv::flip(renderedImg, renderedImg, 0);
	int i=round(displayPt[0]);
	int j=round(displayPt[1]);
	renderedImg.at<cv::Vec3b>(j,i)[0] = 0;
	renderedImg.at<cv::Vec3b>(j,i)[1] = 0;
	renderedImg.at<cv::Vec3b>(j,i)[2] = 255;
	cv::flip(renderedImg, renderedImg, 0);
}

void obj_scan::showdepth(cv::Mat & renderedImg, vector<vector<double>> const & depthmat){
	cv::flip(renderedImg, renderedImg, 0);
	int px,py,pd;
	vector<double> dis;
	for (int i=0;i<depthmat.size();i++){
		dis.push_back(depthmat[i][3]);
	}
	double maxdis = *max_element(dis.begin(),dis.end());
	double mindis = *min_element(dis.begin(),dis.end());
	cout << "maxdis is: " << maxdis << endl;
	cout << "mindis is: " << mindis << endl;
	for (int i=0;i<depthmat.size();i++){
		px = int(round(depthmat[i][0]));
		py = int(round(depthmat[i][1]));
		renderedImg.at<cv::Vec3b>(py,px)[0] = int(round(255.0*(depthmat[i][3]-mindis)/(maxdis-mindis)));
		renderedImg.at<cv::Vec3b>(py,px)[1] = int(round(255.0*(depthmat[i][3]-mindis)/(maxdis-mindis)));
		renderedImg.at<cv::Vec3b>(py,px)[2] = int(round(255.0*(depthmat[i][3]-mindis)/(maxdis-mindis)));
	}
	cv::flip(renderedImg, renderedImg, 0);
}



void obj_scan::intersect(double *intersection, double *tpt, vector<vector<double>>& scanpc, string mode = "OBBTree"){
	if (mode == "OBBTree"){
		obj_scan::OBBTreeintersect(intersection, tpt, scanpc);
	}
	else if (mode == "CellLocator"){
		obj_scan::CellLocatorintersect(intersection, tpt, scanpc);
	}
}

