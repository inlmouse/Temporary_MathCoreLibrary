using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CaffeSharp;
using Emgu.CV;
using Emgu.CV.Structure;
using FastDetection;

namespace csharp_test
{
    class Program
    {
        static void Main(string[] args)
        {
            string net1 = @"IPBBox_deploy.prototxt";
            string model1 = @"IPBBox_iter_iter_60000.caffemodel";
            CaffeModel IPBbox=new CaffeModel(net1,model1,false);

            string net2 = @"5IPTs_deploy.prototxt";
            string model2 = @"5IPTs_iter_iter_300000.caffemodel";
            CaffeModel IPTs5 = new CaffeModel(net2, model2, false);

            FastFace ff=new FastFace(1.1f,3,24);

            Bitmap bmp=new Bitmap(@"D:\Research\FacialLandmarks\Data\UMD\umdfaces_batch2\stan_freberg\stan_freberg_0013.jpg");


            FaceInfo info = ff.Facedetect_Multiview_Reinforce(bmp);

            Bitmap[] bits = new Bitmap[info.count];
            for (int i = 0; i < info.count; i++)
            {
                //int StartX = Convert.ToInt32(info.r[i].X - info.r[i].Width*0.2);
                //int StartY = Convert.ToInt32(info.r[i].Y - info.r[i].Height * 0.2);
                //int iWidth = Convert.ToInt32(info.r[i].Width * 1.4);
                //int iHeight = Convert.ToInt32(info.r[i].Height * 1.4);
                //bits[i] = KiCut(bmp, StartX, StartY, iWidth, iHeight);

                Image<Bgr, byte> image = new Image<Bgr, byte>(bmp);
                Rectangle rect = info.r[i];
                rect.X = Convert.ToInt32(rect.X - rect.Width * 0.2);
                rect.Y = Convert.ToInt32(rect.Y - rect.Height * 0.2);
                rect.Width = Convert.ToInt32(rect.Width * 1.4);
                rect.Height = Convert.ToInt32(rect.Height * 1.4);
                image.ROI = rect;
                bits[i] = image.Bitmap;
                image.ROI = Rectangle.Empty; ;
            }
            if (info.count>0)
            {
                float[][] aa = IPBbox.ExtractBitmapOutputs(bits, new[] { "fc2", "fc3" }, 0);
                Rectangle[] rects = new Rectangle[info.count];
                Bitmap[] C = CaffeModel.Align_Step1(bits, rects, aa[0], aa[1]);

                float[] bb = IPTs5.ExtractBitmapOutputs(C, "fc2", 0);
                Bitmap[] F = CaffeModel.Align_Step2(bits, C, bb, rects, 128, 128);

                F[0].Save("test.jpg");
                float[][] a =
                IPBbox.ExtractFileOutputs(new[] { @"D:\Research\FacialLandmarks\Data\5PTS\batch1_1.jpg" },
                    new[] { "fc2", "fc3" }, 0);
            }
            
        }

        public static Bitmap KiCut(Bitmap b, int StartX, int StartY, int iWidth, int iHeight)
        {
            if (b == null)
            {
                return null;
            }
            int w = b.Width;
            int h = b.Height;
            if (StartX >= w || StartY >= h)
            {
                return null;
            }
            if (StartX + iWidth > w)
            {
                iWidth = w - StartX;
            }
            if (StartY + iHeight > h)
            {
                iHeight = h - StartY;
            }
            try
            {
                Bitmap bmpOut = new Bitmap(iWidth, iHeight, PixelFormat.Format32bppArgb);
                Graphics g = Graphics.FromImage(bmpOut);
                g.DrawImage(b, new Rectangle(0, 0, iWidth, iHeight), new Rectangle(StartX, StartY, iWidth, iHeight), GraphicsUnit.Pixel);
                g.Dispose();
                return bmpOut;
            }
            catch
            {
                return null;
            }
        }
    }
}
