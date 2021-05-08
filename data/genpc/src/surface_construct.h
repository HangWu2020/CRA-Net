/*
 * surface_construct.h
 *
 *  Created on: 2020年4月14日
 *      Author: wuhang
 */

#ifndef SURFACE_CONSTRUCT_H_
#define SURFACE_CONSTRUCT_H_

#include <vtkSmartPointer.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyData.h>
#include <vtkSurfaceReconstructionFilter.h>
#include <vtkContourFilter.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include <vtkOBJExporter.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/obj_io.h>
using namespace std;

class surface_construct{
private:
	vtkSmartPointer<vtkPolyDataReader> reader = vtkSmartPointer<vtkPolyDataReader>::New();
	vtkSmartPointer<vtkPolyData> points = vtkSmartPointer<vtkPolyData>::New();
	vtkSmartPointer<vtkSurfaceReconstructionFilter> surf = vtkSmartPointer<vtkSurfaceReconstructionFilter>::New();
	vtkSmartPointer<vtkContourFilter> contour = vtkSmartPointer<vtkContourFilter>::New();
	vtkSmartPointer <vtkVertexGlyphFilter> vertexGlyphFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
	vtkSmartPointer<vtkActor> pointActor = vtkSmartPointer<vtkActor>::New();
	vtkSmartPointer<vtkPolyDataMapper> vertexMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	vtkSmartPointer<vtkActor> vertexActor = vtkSmartPointer<vtkActor>::New();


public:
	vtkPolyData *data;
	pcl::PolygonMesh pcldata;
	surface_construct(string const& filename);
	void savepoly(string const& objfilename);
};



#endif /* SURFACE_CONSTRUCT_H_ */
