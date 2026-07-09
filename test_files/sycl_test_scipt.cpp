/* #include <sycl/sycl.hpp>
#include <iostream>

int main() {
    sycl::queue q;

    int x = 0;

    {
        sycl::buffer<int> buf(&x, 1);

        q.submit([&](sycl::handler& h) {
            auto acc = buf.get_access<sycl::access::mode::write>(h);

            h.single_task([=] {
                acc[0] = 42;
            });
        });
    }

    std::cout << x << std::endl;
} */

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include <boost/geometry/geometry.hpp>

constexpr size_t N = 1000;
constexpr size_t NUM_MULTS = 1000;

int main() {
    sycl::queue q;
    boost::geometry::model::point<float, 2, boost::geometry::cs::cartesian> p1(1.0f, 2.0f);
    std::cout << "Using device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t iter = 0; iter < NUM_MULTS; ++iter) {
        {
            sycl::buffer<float, 2> a_buf(A.data(), sycl::range<2>(N, N));
            sycl::buffer<float, 2> b_buf(B.data(), sycl::range<2>(N, N));
            sycl::buffer<float, 2> c_buf(C.data(), sycl::range<2>(N, N));

            q.submit([&](sycl::handler& h) {
                auto a = a_buf.get_access<sycl::access::mode::read>(h);
                auto b = b_buf.get_access<sycl::access::mode::read>(h);
                auto c = c_buf.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<2>(N, N),
                               [=](sycl::id<2> idx ){
                    size_t row = idx[0];
                    size_t col = idx[1];

                    float sum = 0.0f;

                    for (size_t k = 0; k < N; ++k) {
                        sum += a[row][k] * b[k][col];
                    }

                    c[row][col] = sum;
                });
            });

            q.wait();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    double elapsed =
        std::chrono::duration<double>(end - start).count();

    std::cout << "Completed "
              << NUM_MULTS
              << " matrix multiplications\n";
    std::cout << "Time: "
              << elapsed
              << " seconds\n";

    std::cout << "C[0] = "
              << C[0]
              << std::endl;

    return 0;
}
// #include <sycl/sycl.hpp>
// #include <iostream>

// int main()
// {
//     auto platforms = sycl::platform::get_platforms();

//     for (auto &p : platforms)
//     {
//         std::cout
//             << p.get_info<sycl::info::platform::name>()
//             << std::endl;

//         for (auto &d : p.get_devices())
//         {
//             std::cout << "  "
//                       << d.get_info<sycl::info::device::name>()
//                       << std::endl;
//         }
//     }

//     sycl::queue q{sycl::default_selector_v};

//     std::cout << "\nSelected Device:\n"
//               << q.get_device()
//                      .get_info<sycl::info::device::name>()
//               << std::endl;
// }

