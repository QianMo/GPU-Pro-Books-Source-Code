
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "SphericalHarmonics.h"

#define REDIST_FACTOR 1
//#define REDIST_FACTOR 1e-30f

const double sqrt2 = 1.4142135623730950488016887242097;

double rcp[MAX_BANDS];
double fTab[MAX_BANDS * (MAX_BANDS + 1)];
double kTab[MAX_BANDS * (MAX_BANDS + 1) / 2];

double getFactor(const int l, const int m){
	/*
	double f = 1;
	for (int i = l - m + 1; i <= l + m; i++){
		f *= i;
	}

	double pmm = 1;
	for (int i = 0; i <= m; i++){
		pmm *= (1 - 2 * i);
	}

	double x = sqrt(pmm * pmm / f);
	*/

	double x = 1.0;

	int i = l - m + 1;
	int n = l + m;
	int k = 0;

	while (true){
		bool b0 = (k <= m);
		bool b1 = (i <= n);

		if (!b0 && !b1) break;

		if ((x <= 1.0 && b0) || !b1){
			int f = (1 - 2 * k);
			x *= (f * f);
			k++;
		} else {
			x /= i;
			i++;
		}
	}

	x = sqrt(x);

	if (m & 1) x = -x;
	return x;
}

void initSH(){
	long double fact[2 * MAX_BANDS];
	long double d = 1.0L;

	rcp[0] = 0;
	for (int i = 1; i < MAX_BANDS; i++){
		rcp[i] = double(1.0L / (long double) i);
	}

	fact[0] = d;
	for (int i = 1; i < 2 * MAX_BANDS; i++){
		d *= i;
		fact[i] = d;
	}

	double *dstF = fTab;
	double *dstK = kTab;
	for (int l = 0; l < MAX_BANDS; l++){

//		double fct = 1;
//		double pmm = 1;

		for (int m = 0; m <= l; m++){
//			pmm *= fct;
//			fct -= 2;

			if (l != m){
				*dstF++ = double(2 * l - 1) / double(l - m);
				*dstF++ = double(l + m - 1) / double(l - m);
			} else {
				*dstF++ = 0;
				*dstF++ = 0;
			}

//			*dstK++ = (double) (pmm * sqrt((long double(2 * l + 1) * fact[l - m]) / (4.0L * 3.1415926535897932384626433832795L * fact[l + m])) / REDIST_FACTOR);
			*dstK++ = (double) (getFactor(l, m) * sqrt((double(2 * l + 1) / (4.0L * 3.1415926535897932384626433832795L))) / REDIST_FACTOR);
		}
	}


}
/*
float K_list[] = {
	0.28209478785f, 0.4886025051f, 0.34549414466f, 0.63078312173f, 0.2575161311f, 0.12875806555f, 0.7463526548f, 0.21545345308f, 0.068132364148f, 0.027814921189f,
	0.84628436355f, 0.18923493652f, 0.044603102283f, 0.011920680509f, 0.0042145970123f, 0.93560256661f, 0.17081687686f, 0.032281355422f, 0.0065894040825f,
	0.0015531374369f, 0.00049114518199f, 1.0171072221f, 0.15694305165f, 0.024814875307f, 0.0041358125511f, 0.00075509260929f, 0.00016098628522f, 4.6472737553e-005f,
	1.0925484154f, 0.14599792317f, 0.019867800849f, 0.0028097313415f, 0.00042358293734f, 7.0597156223e-005f, 1.384524143e-005f, 3.7002964192e-006f, 1.1631066067f, 0.13707342814f, 0.01638340829f, 0.0020166581537f, 0.00026034944814f, 3.6103972493e-005f, 5.5709639026e-006f, 1.0171141988e-006f, 2.5427854971e-007f, 1.2296226727f, 0.12961361028f, 0.013816857281f, 0.0015075427227f, 0.00017069560029f, 2.0402026496e-005f, 2.6338902949e-006f, 3.801693177e-007f, 6.51985001e-008f, 1.5367433848e-008f, 1.2927207185f, 0.12325608434f, 0.011860322246f, 0.0011630002801f, 0.00011748076923f, 1.2383560401e-005f, 1.384524143e-006f, 1.678982142e-007f, 2.2848052974e-008f, 3.7064436234e-009f, 8.2878598968e-010f, 1.3528790761f, 0.11775300918f, 0.0103276221f, 0.00092005770282f, 8.3989393007e-005f, 7.9362516666e-006f, 7.8580600881e-007f, 8.2831226229e-008f, 9.5013932764e-009f, 1.2266245975e-009f, 1.8927228454e-010f, 4.0352986651e-011f, 1.4104739392f, 0.11292829394f, 0.0091000212546f, 0.00074301362407f, 6.1917802006e-005f, 5.3094077196e-006f, 4.7299963365e-007f, 4.4300474576e-008f, 4.4300474576e-009f, 4.8335780492e-010f, 5.9497232884e-011f, 8.7723884023e-012f, 1.7906562843e-012f, 1.4658075153f, 0.10865288191f, 0.0080985076633f, 0.00061044798366f, 4.6819223098e-005f, 3.6784655714e-006f, 2.9836295628e-007f, 2.5216272196e-008f, 2.2464440745e-009f, 2.1419003838e-010f, 2.2330855175e-011f, 2.6317165207e-012f, 3.7218091959e-013f, 7.2990683522e-014f, 1.5191269238f, 0.10482971704f, 0.0072686330764f, 0.00050890610675f, 3.6166382172e-005f, 2.6237851318e-006f, 1.9556539711e-007f, 1.5088197955e-008f, 1.2158416398e-009f, 1.0349931362e-010f, 9.4481514597e-012f, 9.4481514597e-013f, 1.0697924913e-013f, 1.4558031857e-014f, 2.7512094195e-015f, 1.5706373067f, 0.1013842022f, 0.0065717617374f, 0.00042960950433f, 2.8451584469e-005f, 1.9182054336e-006f, 1.3236875055e-007f, 9.40703748e-009f, 6.9349600383e-010f, 5.3504378288e-011f, 4.3686141937e-012f, 3.8315281119e-013f, 3.6868896447e-014f,
	4.0227263989e-015f, 5.2820985382e-016f, 9.6437484011e-017f, 1.6205111811f, 0.098257923044f, 0.0059797867672f, 0.00036664425087f, 2.2738311173e-005f, 1.4323789666e-006f, 9.2076807318e-008f, 6.0713648798e-009f, 4.131040555e-010f, 2.9210867898e-011f, 2.1652535868e-012f, 1.7011838825e-013f, 1.4377627964e-014f, 1.3349292434e-015f, 1.4071389748e-016f, 1.7870682851e-017f, 3.159120257e-018f, 1.6688952713f, 0.095404392594f, 0.0054718171853f, 0.00031591551249f, 1.8424566844e-005f, 1.0894674765e-006f, 6.5578235786e-008f, 4.0360614093e-009f, 2.552629366e-010f, 1.668706019e-011f, 1.1354106326e-012f, 8.1100759469e-014f, 6.1482327086e-015f, 5.0200109853e-016f, 4.508102946e-017f, 4.6010633023e-018f, 5.6635174197e-019f, 9.7128522441e-020f, 1.7159156055f, 0.092786089356f, 0.0050320322108f, 0.00027451986316f, 1.5111821107e-005f, 8.4214886554e-007f, 4.7677290989e-008f, 2.7526496787e-009f, 1.6276758768e-010f, 9.9057199338e-012f, 6.2400170243e-013f, 4.0967718822e-014f, 2.8270411804e-015f, 2.0728871658e-016f, 1.6387611941e-017f, 1.4263585367e-018f, 1.4123054005e-019f, 1.6880278198e-020f, 2.8133796997e-021f, 1.7616813855f, 0.090372348238f, 0.0046482520257f, 0.00024035539025f, 1.2529390841e-005f, 6.6035687919e-007f, 3.5297559928e-008f, 1.9199341526e-009f, 1.0666300848e-010f, 6.0776889666e-012f, 3.5689418491e-013f, 2.1719888413e-014f, 1.3792142934e-015f, 9.2152631233e-017f, 6.5490023309e-018f, 5.0228554997e-019f, 4.2450876965e-020f, 4.0848375405e-021f, 4.7485271873e-022f, 7.7031282861e-023f, 1.8062879733f, 0.088137828247f, 0.0043109620947f, 0.00021187222805f, 1.0489238301e-005f, 5.2446191507e-007f, 2.6557161397e-008f, 1.3659529897e-009f, 7.1595390337e-011f, 3.8379157708e-012f, 2.1127030986e-013f, 1.1999347271e-014f, 7.0706831878e-016f, 4.3517046791e-017f, 2.8207911768e-018f, 1.9465308412e-019f, 1.4508584271e-020f, 1.1925982807e-021f, 1.1169706286e-022f, 1.2647201906e-023f, 1.9996982025e-024f, 1.8498192293f, 0.08606137925f, 0.0040126324977f, 0.00018790873322f,
	8.8581026337e-006f, 4.2133697531e-007f, 2.0271584676e-008f, 9.8915204822e-010f, 4.9090791891e-011f, 2.4858088755e-012f, 1.2888318631e-013f, 6.8694969119e-015f, 3.7815335923e-016f, 2.1617588692e-017f,	1.2918980235e-018f, 8.1381925936e-020f, 5.4619972377e-021f, 3.9625492741e-022f, 3.1725784981e-023f, 2.8961546815e-024f, 3.1982678122e-025f, 4.935034375e-026f, 1.8923493652f, 0.084125190447f, 0.0037472338125f, 0.00016758139065f, 7.539843222e-006f, 3.4201423397e-007f, 1.5676196712e-008f, 7.2774916245e-010f, 3.4306424518e-011f, 1.6467610741e-012f, 8.0739104336e-014f, 4.0572926514e-015f, 2.0979760847e-016f, 1.1214153878e-017f, 6.2300854879e-019f, 3.6211636357e-020f, 2.2202778798e-021f, 1.4514410557e-022f, 1.026323813e-023f, 8.0142425394e-025f, 7.1396547136e-026f, 7.6988876748e-027f, 1.1606509873e-027f, 1.9339444301f, 0.082314141311f, 0.0035098867787f, 0.00015020928745f, 6.4639785415e-006f, 2.8024901181e-007f, 1.2266166018e-008f, 5.4315496068e-010f, 2.438837769e-011f, 1.1131720502e-012f, 5.1789450806e-014f, 2.4633729657e-015f, 1.2020029285e-016f, 6.0402919859e-018f, 3.1401982755e-019f, 1.6980247563e-020f, 9.6131721747e-022f, 5.7449692048e-023f, 3.6628585823e-024f, 2.5276125564e-025f, 1.927286506e-026f, 1.6774875866e-027f, 1.7682271734e-028f, 2.6071087338e-029f, 1.9746635149f, 0.080615300422f, 0.0032966047858f, 0.00013526133275f, 5.5780833482e-006f, 2.3161730416e-007f, 9.7013813075e-009f, 4.1069221938e-010f, 1.7608283638e-011f, 7.6630218528e-013f, 3.3932431105e-014f, 1.5329109834e-015f, 7.0858835275e-017f, 3.3628115035e-018f, 1.6448048712e-019f, 8.328793221e-021f, 4.3896594565e-022f, 2.4237839636e-023f, 1.4135795137e-024f, 8.8005616488e-026f, 5.9333374532e-027f, 4.4224486263e-028f, 3.7646382751e-029f, 3.8829279014e-030f, 5.6045236728e-031f, 2.0145597375f, 0.079017533944f, 0.0031041018936f, 0.00012231875022f, 4.842645695e-006f, 1.9293562265e-007f, 7.7484810674e-009f, 3.1424240259e-010f, 1.2893521955e-011f, 5.362998122e-013f, 2.2662803405e-014f,
	9.7525177961e-016f, 4.2850103139e-017f, 1.9279172853e-018f, 8.9118008037e-020f, 4.2485343345e-021f, 2.0982015523e-022f, 1.0791984281e-023f, 5.8186479667e-025f, 3.3154823825e-026f, 2.0177383221e-027f, 1.33045725e-028f, 9.7033567732e-030f, 8.0861306443e-031f, 8.1682254459e-032f, 1.1551615206e-032f, 2.0536810547f, 0.077511196458f, 0.0029296478521f, 0.00011104801525f, 4.2275256908e-006f, 1.618803252e-007f, 6.2446691179e-009f, 2.4307341574e-010f, 9.5635956295e-012f, 3.8102277015e-013f, 1.5401932619e-014f, 6.3301553126e-016f, 2.6514102928e-017f, 1.1346988547e-018f, 4.9759836503e-020f, 2.24334668e-021f, 1.0436993321e-022f, 5.0331651141e-024f, 2.5292606351e-025f, 1.3330374005e-026f, 7.4287269985e-028f, 4.4237425515e-029f, 2.8555135383e-030f, 2.0396525274e-031f, 1.6653693149e-032f, 1.6489613353e-033f, 2.2866979406e-034f, 2.0920709389f, 0.076087884417f, 0.0027709573165f, 0.00010118105521f, 3.7094774603e-006f, 1.3673315326e-007f, 5.074643279e-009f, 1.8991375165e-010f, 7.178065106e-012f, 2.7446017901e-013f, 1.0635115913e-014f, 4.1843303907e-016f, 1.6750727509e-017f, 6.8384558696e-019f, 2.8543163317e-020f, 1.2215346231e-021f, 5.3775063301e-023f, 2.4443210591e-024f, 1.1522639975e-025f, 5.6630685934e-027f, 2.9205046332e-028f, 1.5932658987e-029f, 9.2921154198e-031f, 5.8768498016e-032f, 4.114617867e-033f, 3.2943308133e-034f, 3.1997369449e-035f, 4.3542904589e-036f, 2.1297689433f, 0.074740237775f, 0.0026261042693f, 9.2500577646e-005f, 3.2703892858e-006f, 1.1620822308e-007f, 4.1555975734e-009f, 1.4975734135e-010f, 5.4466218462e-012f, 2.0022180112e-013f, 7.4514838584e-015f, 2.8123813477e-016f, 1.0784990145e-017f, 4.2108312072e-019f, 1.6776353093e-020f, 6.8375310661e-022f, 2.8589153611e-023f, 1.2302812869e-024f, 5.4692674337e-026f, 2.5227849463e-027f, 1.2137754732e-028f, 6.1304919139e-030f, 3.2768857649e-031f, 1.8732709079e-032f, 1.1617532993e-033f, 7.9789544184e-035f, 6.26885864e-036f, 5.9771221905e-037f, 7.9872654985e-038f, 2.1668111802f, 0.073461779026f,
	0.002493455253f, 8.4829070365e-005f, 2.896016712e-006f, 9.9332553731e-008f, 3.4272987126e-009f, 1.1910686381e-010f, 4.1746934161e-012f, 1.4778254497e-013f, 5.291463929e-015f, 1.9194157044e-016f, 7.0654668469e-018f, 2.6441845117e-019f, 1.0080858943e-020f, 3.9239690218e-022f, 1.563346679e-023f, 6.3929999849e-025f, 2.6919386093e-026f, 1.1715159077e-027f, 5.2923694051e-029f, 2.4948468632e-030f, 1.2351332459e-031f, 6.4738572649e-033f, 3.6303588837e-034f, 2.2093660583e-035f, 1.4895542926e-036f, 1.1492160865e-037f, 1.0763394811e-038f, 1.4133029781e-039f, 2.2032307254f, 0.072246781595f, 0.0023716167942f, 7.8020464477e-005f, 2.5750590401e-006f, 8.5362375968e-008f, 2.8454125323e-009f, 9.548579253e-011f, 3.229855112e-012f, 1.1026543544e-013f, 3.8045189691e-015f, 1.3285956544e-016f, 4.7031776345e-018f, 1.6905222907e-019f, 6.1811616677e-021f, 2.3035829448e-022f, 8.7695903955e-024f, 3.4187422745e-025f, 1.3685922219e-026f, 5.6439791979e-028f, 2.4066008162e-029f, 1.0656607532e-030f, 4.9260185742e-032f, 2.3922860994e-033f, 1.2304591974e-034f, 6.7734549537e-036f, 4.0479135724e-037f, 2.6807962526e-038f, 2.0323063943e-039f, 1.8708896824e-040f, 2.4153081942e-041f, 2.2390579644f, 0.071090161459f, 0.0022593936472f, 7.1953752141e-005f, 2.2984783003e-006f, 7.3723725857e-008f, 2.3769483209e-009f, 7.7118386915e-011f, 2.5206933316e-012f, 8.3104798115e-014f, 2.7670871021e-015f, 9.3172669241e-017f, 3.1771616508e-018f, 1.0988443339e-019f, 3.8609454325e-021f, 1.3806712679e-022f, 5.0347900687e-024f, 1.8763554744e-025f, 7.1639584907e-027f, 2.809935703e-028f, 1.1358491867e-029f, 4.7492240348e-031f, 2.0629310954e-032f, 9.3576454782e-034f, 4.4610824434e-035f, 2.2531868908e-036f, 1.2183839317e-037f, 7.1545967982e-039f, 4.6572458283e-040f, 3.4713060867e-041f, 3.142772863e-042f, 3.9913255273e-043f,
};*/

inline float factorial(const int x){

	float f = 1.0f;
	for (int i = 2; i <= x; i++){
		f *= i;
	}

	return f;

//	return f[x];
}

float P(const int l, const int m, const float x){
	float pmm = 1.0f;
	if (m > 0){
		float somx2 = sqrtf((1.0f - x) * (1.0f + x));

		float fact = 1.0f;
		for (int i = 1; i <= m; i++){
			pmm *= (-fact) * somx2;
			fact += 2.0f;
		}
	}
	if (l == m) return pmm;

	float pmmp1 = x * (2.0f * m + 1.0f) * pmm;
	if (l == m + 1) return pmmp1;

	float pll = 0.0f;
	for (int ll = m + 2; ll <= l; ll++){
		pll = ((2.0f * ll - 1.0f) * x * pmmp1 - (ll + m - 1.0f) * pmm) / (ll - m);
		pmm = pmmp1;
		pmmp1 = pll;
	}

	return pll;
}

float Pm0(const int l, const float x){
	if (l == 0) return 1.0f;
	if (l == 1) return x;

	float pmm = 1.0f;
	float pmmp1 = x;
	float pll;
	float f = 1.0f;
	for (float ll = 2; ll <= l; ll++){
		f += 2.0f;
		pll = (f * x * pmmp1 - (ll - 1.0f) * pmm) / ll;
		pmm = pmmp1;
		pmmp1 = pll;
	}

	return pll;
}

float PmX(const int l, const int m, const float x, float pmm0){
	if (l == m) return pmm0;

	float f = float(2 * m + 1);
	float pmmp1 = f * x * pmm0;
	if (l == m + 1) return pmmp1;

	float pll;
	float d = 2.0f;
	float f2 = float(2 * m);
	for (float ll = float(m + 2); ll <= l; ll++){
		f += 2.0f;
		f2++;
		pll = (f * x * pmmp1 - f2 * pmm0) / d;
		pmm0 = pmmp1;
		pmmp1 = pll;
		d++;
	}

	return pll;
}

float P2(const int l, const int m, const float x){
	float pmm = 1.0f;
	if (m > 0){
		float somx2 = sqrtf(1.0f - x * x);

		float fact = -1.0f;
		for (int i = 1; i <= m; i++){
			pmm *= fact * somx2;
			fact -= 2.0f;
		}

/*
		for (int i = 1; i <= m; i++){
			pmm *= somx2;
		}
		pmm *= f2[m];
*/
		//pmm *= powf(somx2, m);
	}
	if (l == m) return pmm;

	float pmmp1 = x * (2.0f * m + 1.0f) * pmm;
	if (l == m + 1) return pmmp1;

	float pll;
	for (int ll = m + 2; ll <= l; ll++){
		pll = ((2.0f * ll - 1.0f) * x * pmmp1 - (ll + m - 1.0f) * pmm) / (ll - m);
		pmm = pmmp1;
		pmmp1 = pll;
	}

	return pll;
}

float K(const int l, const int m){
	// renormalisation constant for SH function
	float temp = ((2.0f * l + 1.0f) * factorial(l - m)) / (4.0f * PI * factorial(l + m));
	return sqrtf(temp);
}

inline float K2(const int l, const int m){
	return (float) kTab[((l * (l + 1)) >> 1) + m];
}

// return a point sample of a Spherical Harmonic basis function
// l is the band, range [0..N]
// m in the range [-l..l]
// theta in the range [0..Pi]
// phi in the range [0..2*Pi]
float SH(const int l, const int m, const float theta, const float phi){
	const float sqrt2 = sqrtf(2.0f);
	if (m == 0)
		return K(l, 0) * P(l, m, cosf(theta));
	else if (m > 0)
		return sqrt2 * K(l, m) * cosf(m * phi) * P(l, m, cosf(theta));
	else
		return sqrt2 * K(l, -m) * sinf(-m * phi) * P(l, -m, cosf(theta));
}

float SH2(const int l, const int m, const float cosTheta, const float scPhi){

	const int m2 = abs(m);

	float k = K2(l, m2) * P2(l, m2, cosTheta);

	if (m == 0)
		return k;

	return k * scPhi;
}

float SH(const int l, const int m, const float3 &pos){
	float len = length(pos);

	float p = atan2f(pos.z, pos.x);
//	float t = PI / 2 - asinf(pos.y / len);
	float t = acosf(pos.y / len);

	return SH(l, m, t, p);
}

float SH_A(const int l, const int m, const float3 &pos){
	float d = dot(pos, pos);
	float len = sqrtf(d);

	float p = atan2f(pos.z, pos.x);
	float t = acosf(pos.y / len);

	return SH(l, m, t, p) * powf(d, -1.5f);
}

float SH_A2_pos(const int l, const int m, const float3 &pos){
	float dxz = pos.x * pos.x + pos.z * pos.z;

//	float d = dot(pos, pos);
	float d = dxz + pos.y * pos.y;
	float len = sqrtf(d);

//	float p = atan2f(pos.z, pos.x);
	float t = pos.y / len;

//	float cp = cosf(p);
//	float sp = sinf(p);

	float xzLenInv = 1.0f / sqrtf(dxz);

	float cp = pos.x * xzLenInv;
	float sp = pos.z * xzLenInv;


	float c = (float) sqrt2;
	float s = 0.0f;
	for (int i = 0; i < m; i++){
		float ssp = s * sp;
		float csp = c * sp;

		c = c * cp - ssp;
		s = s * cp + csp;
	}


	float scPhi = c;
//	float scPhi = cosf(m * p);

	return SH2(l, m, t, scPhi) * powf(d, -1.5f);
}

float SH_A2_neg(const int l, const int m, const float3 &pos){
/*
	float d = dot(pos, pos);
	float len = sqrtf(d);

	float p = atan2f(pos.z, pos.x);
	float t = pos.y / len;
*/
	float dxz = pos.x * pos.x + pos.z * pos.z;
	float d = dxz + pos.y * pos.y;
	float len = sqrtf(d);

	float t = pos.y / len;
	float xzLenInv = 1.0f / sqrtf(dxz);

	float cp = pos.x * xzLenInv;
	float sp = pos.z * xzLenInv;

	float c = (float) sqrt2;
	float s = 0.0f;
	for (int i = 0; i < m; i++){
		float ssp = s * sp;
		float csp = c * sp;

		c = c * cp - ssp;
		s = s * cp + csp;
	}


//	float scPhi = sinf(m * p);
	float scPhi = s;

	return SH2(l, m, t, scPhi) * powf(d, -1.5f);
}

float SH_A2(const int l, const int m, const float3 &pos){
	if (m >= 0)
		return SH_A2_pos(l, m, pos);
	else
		return SH_A2_neg(l, -m, pos);
}

template <typename FLOAT>
bool cubemapToSH(FLOAT *dst, const Image &img, const int bands){
	if (!img.isCube()) return false;

	FORMAT format = img.getFormat();
	if (format < FORMAT_I32F || format > FORMAT_RGBA32F) return false;
	
	int size = img.getWidth();
	int sliceSize = img.getSliceSize();

	int nCoeffs = bands * bands;
	float *coeffs = new float[nCoeffs];

	for (int i = 0; i < nCoeffs; i++){
		dst[i] = 0;
	}

	float3 v;
	float *src = (float *) img.getPixels();

	for (v.x = 1; v.x >= -1; v.x -= 2){
		for (int y = 0; y < size; y++){
			for (int z = 0; z < size; z++){
				v.y =  1 - 2 * float(y + 0.5f) / size;
				v.z = (1 - 2 * float(z + 0.5f) / size) * v.x;

				computeSHCoefficients(coeffs, bands, v, true);

				float v = *src++;
				for (int i = 0; i < nCoeffs; i++){
					dst[i] += v * coeffs[i];
				}
			}
		}
	}

	for (v.y = 1; v.y >= -1; v.y -= 2){
		for (int z = 0; z < size; z++){
			for (int x = 0; x < size; x++){
				v.x =  2 * float(x + 0.5f) / size - 1;
				v.z = (2 * float(z + 0.5f) / size - 1) * v.y;

				computeSHCoefficients(coeffs, bands, v, true);

				float v = *src++;
				for (int i = 0; i < nCoeffs; i++){
					dst[i] += v * coeffs[i];
				}
			}
		}
	}

	for (v.z = 1; v.z >= -1; v.z -= 2){
		for (int y = 0; y < size; y++){
			for (int x = 0; x < size; x++){
				v.x = (2 * float(x + 0.5f) / size - 1) * v.z;
				v.y =  1 - 2 * float(y + 0.5f) / size;

				computeSHCoefficients(coeffs, bands, v, true);

				float v = *src++;
				for (int i = 0; i < nCoeffs; i++){
					dst[i] += v * coeffs[i];
				}
			}
		}
	}

	delete coeffs;

	float normFactor = 1.0f / (6 * size * size);
	for (int i = 0; i < nCoeffs; i++){
		dst[i] *= normFactor;
	}

	return true;
}

template <typename FLOAT>
bool shToCubemap(Image &img, const int size, const FLOAT *src, const int bands){
	img.create(FORMAT_I32F, size, size, 0, 1);

	int sliceSize = img.getSliceSize();
//	memset(img.getPixels(), 0, 6 * sliceSize);

	int nCoeffs = bands * bands;
	FLOAT *coeffs = new FLOAT[nCoeffs];

	float *dst = (float *) (img.getPixels());

	FLOAT scale = 1.0f;

	float3 v;
	for (v.x = 1; v.x >= -1; v.x -= 2){
		for (int y = 0; y < size; y++){
			for (int z = 0; z < size; z++){
				v.y =  1 - 2 * float(y + 0.5f) / size;
				v.z = (1 - 2 * float(z + 0.5f) / size) * v.x;

				computeSHCoefficients(coeffs, bands, v, false);

				FLOAT v = 0;
				for (int i = 0; i < nCoeffs; i++){
					v += src[i] * coeffs[i];
				}
				*dst++ = float(v * scale);
			}
		}
	}

	for (v.y = 1; v.y >= -1; v.y -= 2){
		for (int z = 0; z < size; z++){
			for (int x = 0; x < size; x++){
				v.x =  2 * float(x + 0.5f) / size - 1;
				v.z = (2 * float(z + 0.5f) / size - 1) * v.y;

				computeSHCoefficients(coeffs, bands, v, false);

				FLOAT v = 0;
				for (int i = 0; i < nCoeffs; i++){
					v += src[i] * coeffs[i];
				}
				*dst++ = float(v * scale);
			}
		}
	}

	for (v.z = 1; v.z >= -1; v.z -= 2){
		for (int y = 0; y < size; y++){
			for (int x = 0; x < size; x++){
				v.x = (2 * float(x + 0.5f) / size - 1) * v.z;
				v.y =  1 - 2 * float(y + 0.5f) / size;

				computeSHCoefficients(coeffs, bands, v, false);

				FLOAT v = 0;
				for (int i = 0; i < nCoeffs; i++){
					v += src[i] * coeffs[i];
				}
				*dst++ = float(v * scale);
			}
		}
	}

	delete coeffs;

	return true;
}

template <typename FLOAT>
void computeSHCoefficients(FLOAT *dest, const int bands, const float3 &pos, const bool fade){
	FLOAT dxz = pos.x * pos.x + pos.z * pos.z;
	FLOAT dxyz = dxz + pos.y * pos.y;
	FLOAT xzLen = sqrt(dxz);
	FLOAT xyzLen = sqrt(dxyz);
	FLOAT xzLenInv = 1.0f / xzLen;
	FLOAT xyzLenInv = 1.0f / xyzLen;

	FLOAT ct = pos.y * xyzLenInv;
	FLOAT st = xzLen * xyzLenInv;

	FLOAT cp = pos.x * xzLenInv;
	FLOAT sp = pos.z * xzLenInv;

//	FLOAT a = powf(xyzLen, -3.0f);
//	FLOAT a = 1.0f / (dxyz * xyzLen);
	FLOAT a = fade? xyzLenInv * xyzLenInv * xyzLenInv : 1;




	if (bands > 0) dest[0] = FLOAT(kTab[0] * a * REDIST_FACTOR);
	if (bands > 1) dest[2] = FLOAT(kTab[1] * a * REDIST_FACTOR * ct);

	FLOAT pmm0 = FLOAT(1.0 * REDIST_FACTOR);
	FLOAT pmm1 = FLOAT(ct * REDIST_FACTOR);
	for (int l = 2; l < bands; l++){
		int i = l * (l + 1);

//		FLOAT pll = ((2 * l - 1) * ct * pmm1 - (l - 1) * pmm0) * rcp[l];
		FLOAT pll = FLOAT(ct * pmm1 * fTab[i] - pmm0 * fTab[i + 1]);
		pmm0 = pmm1;
		pmm1 = pll;

		dest[i] = FLOAT(kTab[i >> 1] * pll * a);
	}

	FLOAT c = FLOAT(sqrt2 * a); // Start with sqrtf(2) * a instead of 1.0 so that it's baked into sin & cos values below
	FLOAT s = 0;
	double pmm = double(1.0 * REDIST_FACTOR);
//	FLOAT fact = -1;
	for (int m = 1; m < bands; m++){
		// Compute cos(m * phi) and sin(m * phi) iteratively from initial cos(phi) and sin(phi)
		FLOAT ssp = s * sp;
		FLOAT csp = c * sp;
		c = c * cp - ssp;
		s = s * cp + csp;

		pmm *= /*fact * */st;
//		fact -= 2;

		FLOAT pmm0 = (FLOAT) pmm;
		FLOAT f0 = FLOAT(2 * m + 1);
		FLOAT pmm1;

		FLOAT kp = FLOAT(kTab[((m * (m + 1)) >> 1) + m] * pmm0);
		dest[m * m        ] = kp * s;
		dest[m * m + 2 * m] = kp * c;

		if (m + 1 < bands){
			pmm1 = f0 * ct * pmm0;

			FLOAT kp = FLOAT(kTab[(m * m + 5 * m + 2) >> 1] * pmm1);
			dest[m * m + 2 * m + 2] = kp * s;
			dest[m * m + 4 * m + 2] = kp * c;
		}

		int index = 2;
		for (int l = m + 2; l < bands; l++){
//			f0 += 2.0f;
//			f1++;
//			FLOAT pll = (f0 * ct * pmm1 - f1 * pmm0) * rcp[index++];

			int it = l * (l + 1) + 2 * m;
			FLOAT pll = FLOAT(ct * pmm1 * fTab[it] - pmm0 * fTab[it + 1]);
			pmm0 = pmm1;
			pmm1 = pll;

//			pmm1 = (2 * m - 5) * ct * pmm1;
//			FLOAT pll = pmm1;

			int i = l * (l + 1);

			FLOAT kp = FLOAT(kTab[(i >> 1) + m] * pll);
			dest[i + m] = kp * c;
			dest[i - m] = kp * s;
		}
	}

}

template bool cubemapToSH(float *dst, const Image &img, const int bands);
template bool shToCubemap(Image &img, const int size, const float *src, const int bands);
template bool cubemapToSH(double *dst, const Image &img, const int bands);
template bool shToCubemap(Image &img, const int size, const double *src, const int bands);
