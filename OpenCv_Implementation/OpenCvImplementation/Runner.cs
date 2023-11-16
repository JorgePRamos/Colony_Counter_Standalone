using System;
using System.IO;
using System.Xml.Serialization;
using OpenCvSharp;

namespace OpenCvTest
{

    class Runner
    {
        static void Main(string[] args)
        {
            //Get Files from IMG folder
            string current = AppDomain.CurrentDomain.BaseDirectory;
            string initialPath = "Img/plate_";

            //Images for analysis
            int[] imagesToAnalyze = { 1, 2 };
            foreach (int item in imagesToAnalyze)
            {
                string imagePath = initialPath + item.ToString() + ".jpg";
                Console.WriteLine(">> Processing image: " + imagePath);

                //Grey and color image
                Mat greyImage = Cv2.ImRead(imagePath, ImreadModes.Grayscale);
                Mat originalImage = Cv2.ImRead(imagePath, ImreadModes.Color);

                //Get plate mask
                Mat plateMask = Image_Process.getPlateMask(greyImage, originalImage, debug: 1);

                //Apply mask to grey image
                Mat maskedImage = new Mat();
                Cv2.BitwiseAnd(greyImage, plateMask, maskedImage);

                Cv2.ImShow("Applied Mask", maskedImage);
                Cv2.WaitKey(0);

                //Perform watershed on original
                Image_Process.watershedMethod(originalImage, maskedImage, debug: 1);
            }


        }

    }


}



