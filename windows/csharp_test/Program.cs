using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CaffeSharp;

namespace csharp_test
{
    class Program
    {
        static void Main(string[] args)
        {
            string net = @"IPBBox_deploy.dat";
            string model = @"IPBBox_iter_iter_60000.caffemodel";
            CaffeModel IPBbox=new CaffeModel(net,model,true);

            float[][] a =
            IPBbox.ExtractFileOutputs(new[] {@"D:\Research\FacialLandmarks\Data\5PTS\batch1_1.jpg"},
                new[] {"fc2", "fc3"}, 0);
        }
    }
}
