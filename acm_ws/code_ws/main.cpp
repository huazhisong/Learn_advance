#include <iostream>

using namespace std;
class Solution {
public:
	void replaceSpace(char *str,int length) {
        // 空字符串或者字符串为NULL
        if(str == NULL || length < 1){
            return;
        }
        // 计算空格的个数
        int blank_len = 0;
        int i = 0;
        for(;str[i] != '\0';i++){
            if(str[i] == ' '){
                blank_len +=1;
            }
        }
        // 计算新的字符串长度,如果超过字符串空间则返回
        int new_length = i + blank_len * 2;
        if(new_length > length){
            return;
        }else{
            for(int j = i - 1, p = new_length-1;j >= 0; j -- ){
                if(str[j] == ' '){
                    str[p--] = '0';
                    str[p--] = '2';
                    str[p--] = '%';
                }else{
                    str[p--] = str[j];
                }
            }
            str[new_length] = '\0';
        }


	}
};
int main()
{
    Solution s;
    char str[80] = "";
    s.replaceSpace(str,80);
    cout<<str<<endl;
    return 0;
}
