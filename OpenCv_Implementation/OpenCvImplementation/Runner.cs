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
            string initialPath = "Img/";

            //Images for analysis
            Console.WriteLine("Type of image? \n Dark [D] /Light [L]: ");
            //string imageType = Console.ReadLine();
            string imageType = "L";
            string folder = "Dark";
            if (imageType.ToLower() == "l") folder = "Light";

            Console.WriteLine("Image ID: ");
            //string imageId = Console.ReadLine();
            string imageId = "1";

            string imagePath = initialPath+folder+ "/plate_" + imageType.ToLower() + "_" + imageId.ToString() + ".jpg";
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
            Image_Process.watershedMethod(originalImage, maskedImage, debug: 0);



        }

    }




}



