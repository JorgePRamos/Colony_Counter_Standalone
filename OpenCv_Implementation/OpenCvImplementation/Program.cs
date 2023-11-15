using System;
using System.IO;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using NumSharp;
using System.Diagnostics;

namespace OpenCvTest
{
  class Program
  {
    public static void watershedMethod(Mat originalImage, Mat processImage, int debug = 0)
    {


      //Get border of components
      Mat dilate = processImage;
      Mat erode = processImage;

      Cv2.Dilate(processImage, dilate, null);
      Cv2.Erode(dilate, erode, null);
      Mat border = dilate - erode;


      Mat distanceCalc = new Mat();



      //Calculate distance transform of Thresh image
      Mat distanceTransform = new Mat();
      Cv2.DistanceTransform(processImage, distanceTransform, DistanceTypes.L2, DistanceTransformMasks.Mask3);
      if (debug == 1)
      {
        Cv2.ImShow("Distance Transform", distanceTransform);
        Cv2.WaitKey(0);
      }



      Cv2.Normalize(distanceTransform, distanceTransform, 0, 1.0, NormTypes.MinMax);
      if (debug == 1)
      {
        Cv2.ImShow("Distance Normalize", distanceTransform);
        Cv2.WaitKey(0);
      }




      //Dilate
      Mat kernel1 = Mat.Ones(3, 3, MatType.CV_8UC1);
      /*
      Cv2.Dilate(distanceTransform, distanceTransform, kernel1);
      if (debug == 1)
      {
        Cv2.ImShow("Dilated", distanceTransform);
        Cv2.WaitKey(0);
      }
      */
      Cv2.Erode(distanceTransform, distanceTransform, null);
      if (debug == 1)
      {
        Cv2.ImShow("Erode", distanceTransform);
        Cv2.WaitKey(0);
      }

      double max = 0, min = 0;
      Cv2.MinMaxIdx(distanceTransform, out min, out max);
      //Threshold
      /*
      Cv2.Threshold(distanceTransform, distanceTransform, .4, 1, ThresholdTypes.Binary);
      //Cv2.Threshold(distanceTransform, distanceTransform, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
      */
      Mat sureBackground = new Mat();
      Cv2.Dilate(distanceTransform, sureBackground, kernel1, iterations: 3);

      Cv2.Threshold(distanceTransform, distanceTransform, 0.7 * max, 255, ThresholdTypes.Binary);
      if (debug == 1)
      {
        Cv2.ImShow("Cool Thresholded", distanceTransform);
        Cv2.WaitKey(0);
      }

      //Convert to 8U for watershed
      Mat distanceTransform_8u = new Mat();
      distanceTransform.ConvertTo(distanceTransform_8u, MatType.CV_8UC1);
      int numComponentsFG = Cv2.ConnectedComponents(distanceTransform_8u, out _, PixelConnectivity.Connectivity8);
      Console.WriteLine("Connected components of sure_fg:  " + numComponentsFG);
      //When convert to CV_8UC1 preview is blank
      /*
        if (debug == 1)

        {
          Cv2.ImShow("CV_8UC1 Dilated", distanceTransform_8u);
          Cv2.WaitKey(0);
        }*/


      //Get all markers
      Point[][] contours;
      HierarchyIndex[] hierarchyIndices;
      Cv2.FindContours(distanceTransform_8u, out contours, out hierarchyIndices, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

      //Marker Image for watershed
      Mat markers = Mat.Zeros(distanceTransform.Size(), MatType.CV_32SC1);

      Console.WriteLine("<  Stats  >");
      Console.WriteLine("- Contours num: " + contours.Length);




      //Draw Markers
      Scalar[] randomColors = new Scalar[100];

      // Generate random RGB values and fill the array with random colors
      Random rand = new Random();
      for (int i = 0; i < randomColors.Length; i++)
      {
        int red = rand.Next(130);    // Random value for red (0-255)
        int green = rand.Next(130);  // Random value for green (0-255)
        int blue = rand.Next(130);   // Random value for blue (0-255)

        // Create a Color object with the random RGB values
        randomColors[i] = new Scalar(red, green, blue);
      }
      for (int i = 0; i < contours.Length; i++)
      {
        Cv2.DrawContours(markers, contours, i, randomColors[i], -1);
      }

      //Background Markers
      Cv2.Circle(markers, 5, 5, 3, Scalar.White, -1);


      Mat markersUnit8 = new Mat();
      markers.ConvertTo(markersUnit8, MatType.CV_8U);
      if (debug == 1)
      {
        Cv2.ImShow("Markers", markersUnit8);
        Cv2.WaitKey(0);
      }

      //8 bit conversion
      Cv2.CvtColor(originalImage, originalImage, ColorConversionCodes.BGR2RGB);
      if (debug == 1)
      {
        Cv2.ImShow("Original", originalImage
        );
        Cv2.WaitKey(0);
      }



      //Watershed
      Cv2.Watershed(originalImage, markers);

      markers.ConvertTo(markersUnit8, MatType.CV_8U);
      if (debug == 1)
      {
        Cv2.ImShow("Watershed", markersUnit8);
        Cv2.WaitKey(0);
      }

      Cv2.BitwiseNot(markersUnit8, markersUnit8);
      if (debug == 1)
      {
        Cv2.ImShow("Bitwise Watershed", markersUnit8);
        Cv2.WaitKey(0);
      }
      int numCircles = Cv2.ConnectedComponents(markersUnit8, out _, PixelConnectivity.Connectivity8);
      Console.WriteLine("---> ConnectedComponents Num of circles bitwise: " + numCircles);



    }







    public static Mat getPlateMask(Mat procImage, Mat originalImage, int debug = 0)
    {

      if (debug == 1)
      {
        Cv2.ImShow("Grey Image", procImage);
        Cv2.WaitKey(0);
      }
      //Binary conversion
      double temp = Cv2.Threshold(procImage, procImage, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
      if (debug == 1)
      {
        Cv2.ImShow("Threshold Image", procImage);
        Cv2.WaitKey(0);
      }

      //Remove noise
      Mat element = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(2, 2));
      Cv2.MorphologyEx(procImage, procImage, MorphTypes.Open, element);
      if (debug == 1)
      {
        Cv2.ImShow("DeNoise Image", procImage);
        Cv2.WaitKey(0);
      }

      //Find disk 
      //dp The inverse ratio of resolution. | min_dist = Minimum distance between detected centers. | param1 threshold internal Canny pram2 threshold center
      CircleSegment[] circles = Cv2.HoughCircles(procImage, HoughModes.Gradient, 1, procImage.Size().Width, minRadius: 100, param2: 10);

      if (circles.Any())
      {

        Console.WriteLine($"Existing circles---> {circles.Count()}");
        circles.OrderDescending();
        Mat outputMask = originalImage;
        Mat test = Mat.Zeros(procImage.Size().Height, procImage.Size().Width, MatType.CV_8UC1);

        foreach (CircleSegment c in circles)
        {
          CircleSegment item = c;
          //Crop the image disk
          NDArray plateMask = np.zeros((procImage.Size().Width, procImage.Size().Height), np.uint8);
          Scalar colorScalar = new Scalar(255);
          Console.WriteLine("Radius ---> " + (int)item.Radius);
          Cv2.Circle(test, (int)item.Center.X, (int)item.Center.Y, (int)item.Radius - 25, colorScalar, thickness: -1);
          //output --> outputMask (Mask of the plate)
          if (debug == 1)
          {
            Cv2.ImShow("Mask", test);
            Cv2.WaitKey(0);
          }


        }

        return test;
      }
      Console.WriteLine("No existing circles");
      return null;


    }


    static void Main(string[] args)
    {
      Console.WriteLine("## Start ##");
      string current = AppDomain.CurrentDomain.BaseDirectory;
      int amountToAnalyze = 1;
      string initialPath = "Img/plate_";
      for (int i = 1; i <= i + amountToAnalyze; i++)
      {
        string imagePath = initialPath + i.ToString() + ".jpg";
        Console.WriteLine("---->  " + imagePath);
        Mat greyImage = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
        Mat originalImage = Cv2.ImRead(imagePath, ImreadModes.Color);
        Cv2.ImShow("Grey", greyImage);
        Mat plateMask = getPlateMask(greyImage, originalImage, debug: 0);
        Mat maskedImage = greyImage;
        Cv2.BitwiseAnd(greyImage, plateMask, maskedImage);
        Cv2.ImShow("Applied Mask", maskedImage);

        Cv2.WaitKey(0);

        watershedMethod(originalImage, maskedImage, debug: 1);
        break;
      }



    }
  }
}

