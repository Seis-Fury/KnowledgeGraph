#!/usr/bin/env python
# vim:set fileencoding:utf-8

import os
import subprocess
from subprocess import PIPE,Popen

class FnlpNER():
    def __init__(self,java_option='-Xmx1024m'):
        self._basedir = '/Users/rich/Arch/fnlp'
        self._java_option = java_option
        self._path_to_jar = os.path.join(self._basedir,'fnlp-core/target/fnlp-core-2.1-SNAPSHOT.jar:libs/trove4j-3.0.3.jar:libs/commons-cli-1.2.jar')
        self._path_to_model_seg = os.path.join(self._basedir,'models/seg.m')
        self._path_to_model_pos = os.path.join(self._basedir,'models/pos.m')

    def Ner_f(self,input_file_path,output_file_path):
        """
        """
        cmd = [
                'java',self._java_option,
                '-Dfile.ecoding=UTF-8',
                '-cp',self._path_to_jar,
                'org.fnlp.nlp.cn.tag.NERTagger',
                '-f',self._path_to_model_seg,self._path_to_model_pos,
                input_file_path,output_file_path]
        #p=subprocess.check_output(cmd)
        p =  Popen(cmd, stdin=PIPE,stdout=PIPE,stderr=PIPE)
        #print 'Run'
        return p
    def Ner_s(self,input_file_path=None,output_file_path=None):
        """
        """
        cmd = [
                'java',self._java_option,
                '-Dfile.ecoding=UTF-8',
                '-cp',self._path_to_jar,
                'org.fnlp.nlp.cn.tag.NERTagger',
                '-s',self._path_to_model_seg,self._path_to_model_pos]
        cmd_all = ' '.join(cmd)
        Sent = ''
        if input_file_path is None:
            S = '周杰伦'
            Sent = S.decode('UTF-8')
            cmd_all = cmd_all +" \""+Sent+"\""
            print cmd_all
            p = Popen(cmd_all, stdin=PIPE,stderr=PIPE,shell=True)
            p.wait()
            return p
        else:
            with open(input_file_path,'r') as fp:
                for i in fp:
                    Sent = i.decode('UTF-8')
                    cmd_i = cmd_all +" \""+Sent+"\"" +" >>" + output_file_path
                    print cmd_i
                    p = Popen(cmd_i, stdin=PIPE,stderr=PIPE,shell=True)
                    p.wait()
            return p




if __name__ == '__main__':
    Fd = FnlpNER()
    Inf = './Input.txt'
    Ouf = './2.txt'
    if os.path.exists(Ouf):
        os.remove(Ouf)
    Fd.Ner_s(Inf,Ouf)
    #Fd.Ner_f(Inf,Ouf)
