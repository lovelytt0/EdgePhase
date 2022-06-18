#!/usr/bin/perl -w

$dir = "../phase_30day_sel/20201*";
my @files = glob( $dir );
 
foreach (@files ){
    print $_ . "\n";
    $year = substr($_,19,4);
    $mon = substr($_,23,2);
    $day = substr($_,25,2);
    $fileExist = "../REAL_result_30day_sel/catalog_sel_$year$mon$day.txt";

    if ( -e $fileExist ) {
        print "$fileExist File Exists\n"
    }
    else {    
        $D = "$year/$mon/$day";
        $R = "0.55/30/0.03/5/5/360/180/37.9175/26.7901";
        $G = "4/30/0.01/2";
        $V = "6.2/3.3";
        $S = "5/0/10/1/0.5/0.5/1.3/1.8";
        $dir = "../phase_30day_sel/$year$mon$day";
        $station = "./station.dat";
        $ttime = "./ttdb_small.txt";
#         print"REAL -D$D -R$R -S$S -V$V $station $dir \n";

#         system("REAL -D$D -R$R -S$S -V$V $station $dir");
        
        system("REAL -D$D -R$R -G$G -S$S -V$V $station $dir $ttime");
        print"REAL -D$D -R$R -G$G -S$S -V$V $station $dir $ttime\n";
        use File::Copy qw(copy);
        $old_file = "./phase_sel.txt";
        $new_file = "../REAL_result_30day_sel/phase_sel_$year$mon$day.txt";
        copy $old_file, $new_file;
        $old_cat = "./catalog_sel.txt";
        $new_cat = "../REAL_result_30day_sel/catalog_sel_$year$mon$day.txt";
        copy $old_cat, $new_cat;
    }
}
 

$year = "2020";
$mon = "10";
$day = "29";

$D = "$year/$mon/$day";
$R = "0.2/20/0.02/2/5";
$G = "1.4/20/0.01/1";
$V = "6.2/3.3";
$S = "5/0/15/1/0.5/0.5/1.3/1.8";

$dir = "../Data2/$year$mon$day";
$station = "./station.dat";
$ttime = "./ttdb.txt";

system("REAL -D$D -R$R -G$G -S$S -V$V $station $dir $ttime");
print"REAL -D$D -R$R -G$G -S$S -V$V $station $dir $ttime\n";

use File::Copy qw(copy);
$old_file = "./phase_sel.txt";
$new_file = "../REAL_result/phase_sel_$year$mon$day.txt";

copy $old_file, $new_file;

$old_cat = "./catalog_sel.txt";
$new_cat = "../REAL_result/catalog_sel_$year$mon$day.txt";

copy $old_cat, $new_cat;
