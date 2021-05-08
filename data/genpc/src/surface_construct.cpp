/*
 * surface_construct.cpp
 *
 *  Created on: 2020年4月14日
 *      Author: wuhang
 */

# include "surface_construct.h"

using namespace std;

surface_construct::surface_construct(string const& filename){
	reader->SetFileName(filename.c_str());
	reader->Update();

	points->SetPoints(reader->GetOutput()->GetPoints());

	surf->SetInputData(points);
	surf->SetNeighborhoodSize(20);
	surf->SetSampleSpacing(0.005);
	surf->Update();

	contour->SetInputConnection(surf->GetOutputPort());
	contour->SetValue(0, 0.0);
	contour->Update();

	data = contour->GetOutput();

	vertexGlyphFilter->AddInputData(points);
	vertexGlyphFilter->Update();

	vertexMapper->SetInputData(vertexGlyphFilter->GetOutput());
	vertexMapper->ScalarVisibilityOff();

	vertexActor->SetMapper(vertexMapper);
	vertexActor->GetProperty()->SetColor(1.0, 0.0, 0.0);
}

void surface_construct::savepoly(string const& objfilename){
	pcl::io::vtk2mesh(data, pcldata);
	pcl::io::saveOBJFile(objfilename, pcldata);
}





