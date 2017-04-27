using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using CaffeSharp;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using FastDetection;
using Microsoft.ProjectOxford.Face;
using Newtonsoft.Json;

namespace csharp_test
{
    class Program
    {
        public const string SubscriptionKey = "6987fb28d0114ababb859dbbd1db314c";
        static void Main(string[] args)
        {
            //CaffeModel temp = new CaffeModel(@"C:\Users\BALTHASAR\Desktop\Feature.prototxt", @"C:\Users\BALTHASAR\Desktop\Feature.caffemodel");
            CaffeModel resnet101 = new CaffeModel(@"C:\Research\FaceRecognition\model\centerloss_deploy.prototxt", @"C:\Research\FaceRecognition\snapshot\centerloss_train_iter_50000.caffemodel");
            resnet101.GetUsefulPart(@"C:\Research\FaceRecognition\snapshot\resnet34_usefulpart.caffemodel");
            //SortedList<int, float> results = lfw_fe(@"D:\Research\FaceRecognition\snapshot\GlasssixNet_train_iter_30000.caffemodel",
            //    "GlasssixNet.prototxt", "lfw_pairs.txt", @"D:\Alignment_lfw_Equalized\");
            //float best_th;
            //float ACC = lfw_acc(results, out best_th);
            //lfw_error(best_th, results);
            //Console.WriteLine(ACC);
            //getTestBenchamrkRes(resnet101, "Wrong");
            //getTestBenchamrkRes(resnet101, "Correct");
        }

        static void getTestBenchamrkRes(CaffeModel model, string iscorrect)
        {
            string fatherDir = @"C:\Data\Alignment_TestBenchmark_Equalized\"+iscorrect;
            int count = 0;
            DirectoryInfo fatherInfo = new DirectoryInfo(fatherDir);
            DirectoryInfo[] childrenInfos = fatherInfo.GetDirectories();
            foreach (var childrenInfo in childrenInfos)
            {
                FileInfo[] files = childrenInfo.GetFiles("*.jpg", SearchOption.AllDirectories);
                if (files.Length!=2)
                {
                    continue;
                }
                else
                {
                    Bitmap[] bits1 = new[] {new Bitmap(files[0].FullName)};
                    Bitmap[] bits2 = new[] {new Bitmap(files[1].FullName)};
                    float[] a = model.ExtractBitmapOutputs(bits1, "eltmax_eltwise_pool5", 0);
                    float[] b = model.ExtractBitmapOutputs(bits2, "eltmax_eltwise_pool5", 0);
                    float confidency = CaffeModel.CosineDistanceProb(a, b);
                    Write(@"C:\Users\BALTHASAR\Documents\MATLAB\FaceTest\" + iscorrect + ".txt", confidency.ToString());
                }
                count++;
                Console.WriteLine(count);
            }
        }

        static async void XiaoKaXiu()
        {
            string model1 = @"IPBBox_iter_iter_60000.caffemodel";
            string net1 = @"IPBBox_deploy.prototxt";
            string model2 = @"5IPTs_iter_iter_300000.caffemodel";
            string net2 = @"5IPTs_deploy.prototxt";
            var IPBBox = new CaffeModel(net1, model1, false);
            var IPTs = new CaffeModel(net2, model2, false);
            var ff = new FastFace(1.1f, 3, 48);
            string fatherDir = @"D:\CASIA\Cleaned_CASIA";
            FaceServiceClient faceServiceClient = new FaceServiceClient(SubscriptionKey);
            
            int count = 0;
            //Random rd=new Random(45985369);
            DirectoryInfo fatherInfo = new DirectoryInfo(fatherDir);
            DirectoryInfo[] childrenInfos = fatherInfo.GetDirectories();
            foreach (var childrenInfo in childrenInfos)
            {
                //DirectoryInfo[] SubInfos = childrenInfo.GetDirectories();
                
                FileInfo[] files = childrenInfo.GetFiles("*.jpg", SearchOption.AllDirectories);

                
                int filecounter = 0;
                foreach (var file in files)
                {
                    filecounter++;
                    try
                    {
                        if (count <= -1)
                        {
                            continue;
                        }
                        Bitmap bmp = new Bitmap(file.FullName);
                        FaceInfo info = ff.Facedetect_Frontal_Surveillance(bmp);
                        if (info.count != 1)
                        {
                            //Write("lost.txt", file.FullName);
                            continue;
                        }

                        ////**************
                        Bitmap[] bits = new Bitmap[1];
                        bits[0] = bmp;
                        Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
                        int StartX = Convert.ToInt32(0 - bmp.Width * 0.2);
                        int StartY = Convert.ToInt32(0 - bmp.Height * 0.2);
                        int iWidth = Convert.ToInt32(bmp.Width * 1.4);
                        int iHeight = Convert.ToInt32(bmp.Height * 1.4);
                        bits[0] = KiCut(bmp, StartX, StartY, iWidth, iHeight);

                        float[][] feature = IPBBox.ExtractBitmapOutputs(bits, new[] { "fc2", "fc3" }, 0);
                        if (Math.Abs(feature[1][0] * 90) >= 30)
                        {
                            bits[0].Dispose();
                            bmp.Dispose();
                            continue;
                        }

                        Rectangle[] rects = new Rectangle[bits.Length];
                        Bitmap[] C = CaffeModel.Align_Step1(bits, rects, feature[0], feature[1]);
                        float[] bb = IPTs.ExtractBitmapOutputs(C, "fc2", 0);
                        Bitmap[] F = CaffeModel.Align_Step2(bits, C, bb, rects, 128, 128);

                        if (!Directory.Exists(@"C:\Data\Alignment_CASIA\" + childrenInfo.Name))
                        {
                            Directory.CreateDirectory(@"C:\Data\Alignment_CASIA\" + childrenInfo.Name);
                        }
                        Image<Gray, byte> gray = new Image<Gray, byte>(F[0]);
                        //Image<Gray, byte> temp = gray.Resize(32, 32, Inter.Linear);
                        //temp = temp.Resize(64, 64, Inter.Linear);
                        gray._EqualizeHist();
                        Image<Bgr, byte> colorimg = new Image<Bgr, byte>(new Image<Gray, byte>[] { gray, gray, gray });
                        colorimg.Save(@"C:\Data\Alignment_CASIA\" + childrenInfo.Name + "\\" + filecounter + ".jpg");
                        //temp._EqualizeHist();
                        //temp.Save(@"C:\Data\Alignment_Registed_Equalized\" + childrenInfo.Name + "\\" + filecounter + "_per_blur.jpg");
                        //Console.WriteLine(@"C:\Data\Alignment_CASIA_Equalized\" + childrenInfo.Name + "\\" + filecounter + " complete!");
                        ////*****
                        colorimg.Dispose();
                        gray.Dispose();
                        F[0].Dispose();
                        C[0].Dispose();
                        bits[0].Dispose();
                        bmp.Dispose();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex.ToString());
                        
                        continue;
                        throw;
                    }
                    
                    //}
                    //Console.WriteLine(filecounter);

                }
                Console.WriteLine(count);
                count++;
            }
        }

        private static float[] ImageAligement(Bitmap bit, float[][] feature, int j, CaffeModel caffe5IPTs, int deviceId)
        {
            try
            {
                Image<Bgr, byte> image = new Image<Bgr, byte>(bit);
                Rectangle draw = new Rectangle(Convert.ToInt32(feature[0][j * 4 + 0] * image.Width),
                Convert.ToInt32(feature[0][j * 4 + 1] * image.Height),
                Convert.ToInt32(feature[0][j * 4 + 2] * image.Width - feature[0][j * 4 + 0] * image.Width),
                Convert.ToInt32(feature[0][j * 4 + 3] * image.Height - feature[0][j * 4 + 1] * image.Height));

                //image.Draw(draw, new Bgr(Color.Aqua), image.Width/64);
                //
                int Margin_Height = Convert.ToInt32(draw.Height * 1.2f);
                int Margin_X = Convert.ToInt32(draw.X - draw.Width * 0.1f);
                int Margin_Y = Convert.ToInt32(draw.Y - draw.Height * 0.1f);
                Rectangle Margindraw = new Rectangle(Margin_X, Margin_Y, Margin_Height, Margin_Height);//What's you really need
                //image.Draw(Margindraw, new Bgr(Color.Crimson), image.Width / 64);
                //*****************
                image.ROI = Margindraw;
                Image<Bgr, byte> C = new Image<Bgr, byte>(image.Bitmap);
                float[] pts5 = caffe5IPTs.ExtractBitmapOutputs(new[] { C.Bitmap }, "fc2", deviceId);

                for (int i = 0; i < pts5.Length; i += 2)
                {
                    pts5[i] = (pts5[i] * Margindraw.Width - Margindraw.Width / 12.0f + draw.X - image.Width / 7.0f) / image.Width / 1.4f;
                    pts5[i + 1] = (pts5[i + 1] * Margindraw.Height - Margindraw.Height / 12.0f + draw.Y - image.Height / 7.0f) / image.Height / 1.4f;
                }
                return pts5;
            }
            catch (Exception)
            {
                return null;
            }
        }
        void whatever()
        {
            string net1 = @"IPBBox_deploy.prototxt";
            string model1 = @"IPBBox_iter_iter_60000.caffemodel";
            CaffeModel IPBbox = new CaffeModel(net1, model1, false);

            string net2 = @"5IPTs_deploy.prototxt";
            string model2 = @"5IPTs_iter_iter_300000.caffemodel";
            CaffeModel IPTs5 = new CaffeModel(net2, model2, false);

            FastFace ff = new FastFace(1.1f, 3, 24);

            Bitmap bmp = new Bitmap(@"D:\小咖秀\20153月296月10\20170115150220800-315.jpg");


            FaceInfo info = ff.Facedetect_Multiview_Reinforce(bmp);

            //Bitmap[] bits = new Bitmap[info.count];
            Bitmap[] bits = new[] { bmp };
            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            int StartX = Convert.ToInt32(rect.X - rect.Width * 0.2);
            int StartY = Convert.ToInt32(rect.Y - rect.Height * 0.2);
            int iWidth = Convert.ToInt32(rect.Width * 1.4);
            int iHeight = Convert.ToInt32(rect.Height * 1.4);
            bits[0] = KiCut(bmp, StartX, StartY, iWidth, iHeight);
            for (int i = 0; i < info.count; i++)
            {
                //int StartX = Convert.ToInt32(info.r[i].X - info.r[i].Width * 0.2);
                //int StartY = Convert.ToInt32(info.r[i].Y - info.r[i].Height * 0.2);
                //int iWidth = Convert.ToInt32(info.r[i].Width * 1.4);
                //int iHeight = Convert.ToInt32(info.r[i].Height * 1.4);
                //bits[i] = KiCut(bmp, StartX, StartY, iWidth, iHeight);

                //Image<Bgr, byte> image = new Image<Bgr, byte>(bmp);
                //Rectangle rect = info.r[i];
                //rect.X = Convert.ToInt32(rect.X - rect.Width * 0.2);
                //rect.Y = Convert.ToInt32(rect.Y - rect.Height * 0.2);
                //rect.Width = Convert.ToInt32(rect.Width * 1.4);
                //rect.Height = Convert.ToInt32(rect.Height * 1.4);
                //image.ROI = rect;
                //bits[i] = image.Bitmap;
                //image.ROI = Rectangle.Empty; ;
            }
            if (/*info.count>0*/true)
            {
                float[][] aa = IPBbox.ExtractBitmapOutputs(bits, new[] { "fc2", "fc3" }, 0);
                Rectangle[] rects = new Rectangle[info.count];
                Bitmap[] C = CaffeModel.Align_Step1(bits, rects, aa[0], aa[1]);

                float[] bb = IPTs5.ExtractBitmapOutputs(C, "fc2", 0);
                Bitmap[] F = CaffeModel.Align_Step2(bits, C, bb, rects, 128, 128);

                F[0].Save(@"C:\Users\BALTHASAR\Desktop\test.jpg");
                float[][] a =
                IPBbox.ExtractFileOutputs(new[] { @"D:\Research\FacialLandmarks\Data\5PTS\batch1_1.jpg" },
                    new[] { "fc2", "fc3" }, 0);
            }
        }

        public static byte[] ToByteArray(Image image, ImageFormat format)
        {
            using (MemoryStream ms = new MemoryStream())
            {
                image.Save(ms, format);
                return ms.ToArray();
            }
        }

        static SortedList<int, float> lfw_fe(string model,string net,string pair,string imgDir)
        {
            SortedList<int, float> results=new SortedList<int, float>();
            var fe=new CaffeModel(net,model);
            StreamReader sr=new StreamReader(pair, Encoding.Default);
            string line;
            int label = 1;
            while ((line=sr.ReadLine())!=null)
            {
                string[] sArray = line.Split('\t');
                if (sArray.Length==3)
                {
                    DirectoryInfo child=new DirectoryInfo(imgDir+sArray[0]);
                    FileInfo[] file = child.GetFiles("*.jpg", SearchOption.AllDirectories);
                    float[] a = fe.ExtractFileOutputs(new[] { file[Convert.ToInt32(sArray[1]) - 1].FullName }, "eltmax_fc5", 0);
                    float[] b = fe.ExtractFileOutputs(new[] { file[Convert.ToInt32(sArray[2]) - 1].FullName }, "eltmax_fc5", 0);
                    float confidency = CaffeModel.CosineDistanceProb(a, b);
                    results.Add(label, confidency);
                }
                if (sArray.Length == 4)
                {
                    DirectoryInfo child = new DirectoryInfo(imgDir + sArray[0]);
                    FileInfo[] file = child.GetFiles("*.jpg", SearchOption.AllDirectories);
                    float[] a = fe.ExtractFileOutputs(new[] { file[Convert.ToInt32(sArray[1]) - 1].FullName }, "eltmax_fc5", 0);
                    //
                    child = new DirectoryInfo(imgDir + sArray[2]);
                    file = child.GetFiles("*.jpg", SearchOption.AllDirectories);
                    float[] b = fe.ExtractFileOutputs(new[] { file[Convert.ToInt32(sArray[3]) - 1].FullName }, "eltmax_fc5", 0);
                    float confidency = CaffeModel.CosineDistanceProb(a, b);
                    results.Add(-1 * label, confidency);
                }
                Console.WriteLine(label);
                label++;
            }
            return results;
        }

        static float lfw_acc(SortedList<int, float> s, out float best_th)
        {
            float best_acc = 0;
            best_th = -1;
            for (float i = -1; i <= 1; i=i+0.01f)
            {
                int count = 0;
                for (int j = 0; j < s.Count; j++)
                {
                    if (s.Keys[j] < 0 && s.Values[j] <= i)
                    {
                        count++;
                    }
                    else if (s.Keys[j] > 0 && s.Values[j] > i)
                    {
                        count++;
                    }
                }
                if (count*1.0f/6000>best_acc)
                {
                    best_acc = count*1.0f/6000;
                    best_th = i;
                }
            }

            return best_acc;
        }

        static void lfw_error(float best_th, SortedList<int, float> s)
        {
            int count = 0;
            for (int j = 0; j < s.Count; j++)
            {
                if (s.Keys[j] < 0 && s.Values[j] <= best_th)
                {
                    count++;
                }
                else if (s.Keys[j] > 0 && s.Values[j] > best_th)
                {
                    count++;
                }
                else
                {
                    Console.WriteLine(s.Keys[j] + " " + s.Values[j]);
                }
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
                g.DrawImage(b, new Rectangle(0, 0, iWidth, iHeight), new Rectangle(StartX/2, StartY/2, iWidth, iHeight), GraphicsUnit.Pixel);
                g.Dispose();
                return bmpOut;
            }
            catch
            {
                return null;
            }
        }

        public static void Write(string path, string label)
        {
            FileStream fs = new FileStream(path, FileMode.Append);
            StreamWriter sw = new StreamWriter(fs);
            sw.WriteLine(label);
            sw.Flush();
            sw.Close();
            fs.Close();
        }
    }
}
