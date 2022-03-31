#!/usr/bin/perl -w
$phasein = "../VELEST/phase_allday.txt";
$event = "../VELEST/new.cat";

$phaseout = "italy.pha"; #phase format for hypoDD

open(JK,"<$phasein");
@par = <JK>;
close(JK);

open(EV,"<$event");
@eves = <EV>;
close(EV);

open(EV,">$phaseout");
foreach $file(@par){
    chomp($file);
    ($test,$jk) = split(' ',$file);
    if($test eq "#"){
		($jk,$year,$month,$day,$hour,$min,$sec,$lat,$lon,$dep,$jk,$jk,$jk,$jk,$num) = split(' ',,$file);
			$out = 0;
		foreach $eve(@eves){
			chomp($eve);
			($date,$hh,$mm,$ss,$lat1,$lon1,$dep1,$num0,$gap,$res) = split(" ",$eve);
			$num1 = $num0*100;
            #combine original picks &  improved initial locations that obtained with VELEST
#             print $hour, " ",$hh, " ", $min, " ", $mm, " ", $sec," ", $ss, "\n" ;
			if(abs($num-$num1)<1*2 && abs($hour*3600 + $min*60 + $sec - $hh*3600 - $mm*60 - $ss) < 1.5*4  && $lon1>26 && $lon1 <27.4 &&  $lat1>37.5 && $lat1 <38.3 )
#             if(abs($num-$num1)<1*2 && abs($hour*3600 + $min*60 + $sec - $hh*3600 - $mm*60 - $ss) < 1.5*2  && $lon1>21.2 && $lon1 <23 &&  $lat1>37.5 && $lat1 <39 )
#             if(abs($num-$num1)<1*2 && abs($hour*3600 + $min*60 + $sec - $hh*3600 - $mm*60 - $ss) < 1.5*2  && $lon1>23 && $lon1 <25 &&  $lat1>38 && $lat1 <39 )

            {
				print EV "# $year $month $day $hour $min $sec $lat1 $lon1 $dep1 0 0 0 0 $num\n";
				$out = 1;
			}
		}
	}else{
		if($out>0){
			printf EV "$file\n";
		}
    }
}
close(EV);
