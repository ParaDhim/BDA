#include <iostream>
#include <string>
using namespace std;

string IndianGeek(int N) {
    long long left = 1, right = N * N;
    while (left <= right) {
        long long mid = (left + right) / 2;
        long long count = 0;
        for (int i = 1; i <= N; i++) {
            count += min(mid / i, (long long)N);
        }
        if (count < (N * N + 1) / 2) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return to_string(left);
}

int main() {
    int N;
    cin >> N;
    cout << findMedian(N) << endl;
    return 0;
}