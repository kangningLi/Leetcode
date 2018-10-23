class Solution {
public:
    string convert(string s, int numRows) {
        if (s == || s.length()<2 || numRows < 2)
            return s;
        string res;
        int position = 0;
        int size= 2*numRows -2;
        # The first LINE
        for(int i=0;i< s.length();i+=size){
            res[position++]= s[i];
            }
        # Internal LINE
        for(int i=1;i< numRows-1;i++){
            int inter = (i<<1);
            for(int j=i;j<s.length();j+= inter){
                res[position++]=s[j];
                inter= size-inter;}
              }
        # Last LINE 
        for(int i=numRows-1;i<s.length();i+=size){
            res[position++]=s[i];
        }
        
        
        
        return res;
        
    }
}; 
