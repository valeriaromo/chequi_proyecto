import argparse
import characterize
import kmeans
import prepare
import report
import visualize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full segmentation pipeline")
    parser.add_argument("--k", type=int, default=None, help="Override the K selected by silhouette")
    a = parser.parse_args()
    prepare.main()
    kmeans.main(a.k)
    characterize.main()
    visualize.main()
    print(report.main())