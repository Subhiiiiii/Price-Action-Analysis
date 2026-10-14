"""CLI equivalent of the relaxed BO/SM/W research notebook."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from market_analysis.combined_signal_analysis.relaxed_research import run_relaxed_research


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--as-of",default=None)
    parser.add_argument("--data-mode",choices=["auto","cache"],default="auto")
    parser.add_argument("--front-n",type=int,default=5)
    parser.add_argument("--products",nargs="+",default=["BO","SM","W"])
    args=parser.parse_args()
    result=run_relaxed_research(products=args.products,as_of=args.as_of,data_mode=args.data_mode,front_n=args.front_n)
    print("\nBASELINE\n"+result["baseline"].to_string(index=False))
    print("\nERRORS\n"+result["errors"].to_string(index=False))
    print("Saved:",result["output_dir"])
