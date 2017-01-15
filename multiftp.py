#!/usr/bin/python2.7
from ftplib import FTP
from multiprocessing import Process, Queue
from Queue import Empty
import os
import os.path
import datetime
import argparse

"""
multiftp, by Greg Cordts

multiftp is a multi-threaded (multiprocess) command-line FTP uploader that allows easy recursive uploading of directories. It provides an easy way to schedule fast FTP uploads via batch scripts.

Webpage: http://www.gregcor.com/

For more information, see the README.md file.


This software is licensed under the MIT License:
-----------------------------------------------
Copyright (C) 2011 by Greg Cordts

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""


#The position resulting from a split() of a LIST where the filename starts. I don't know enough about FTP server implementations to know if this is standard or not.
LSVALIDPOS = 8
#Global, gets changed in the command arguments.
SILENT = False

def main():
    """
    Does all setup for the FTP connections. Reads in arguments and passes along correct path

    """
    args = getArgs()

    global SILENT
    SILENT = args["silent"]

    ftpInfo = {
        "server"   : args["host"],
        "port"     : args["port"],
        "user"     : args["user"],
        "password" : args["password"]
        }
    numProcs = args["n"]
    pe = PathExplorer(args["source"])
    dest = args["dest"]
    if not dest.endswith("/"):
        dest += "/"
    if not dest.startswith("/"):
        dest = "/" + dest
    basePath = dest
    fQueue = pe.getFileQueue()

    procs = []
    startTime = datetime.datetime.now()
    i = 1
    for curProc in range(numProcs):
        ul = Uploader(ftpInfo, fQueue, basePath, i)
        ul.start()
        procs.append(ul)
        i += 1
    for curProc in procs:
        curProc.join()
    output("Upload completed successfully!")
    total = (datetime.datetime.now() - startTime).total_seconds()
    output("Total time: %d minutes, %d seconds" % (total / 60, total % 60))

def output(text):
    """
    Output text if the global SILENT is False.
    """
    if not SILENT:
        print(text)

def getArgs():
    """
    Uses argparse, so there's a Python 2.7 minimum
    """
    parser = argparse.ArgumentParser(description="multiftp is a multithreaded (multiprocess), recursive FTP uploader.")
    parser.add_argument("-n", metavar = "num", type=int, default=2,
                        help="Number of concurrent uploads. Default is 2.")
    parser.add_argument("-s", "--silent", default=False, action="store_true",
                        help="Silent mode.")
    parser.add_argument("--port", metavar = "port", type=int, default=21,
                        help="Port on the server. Default is 21.")
    parser.add_argument("--source", metavar="source", type=str, default=".",
                        help="Location to upload from. Defaults to current directory.")
    parser.add_argument("--dest", metavar = "dest", type=str, default="/",
                        help="Destination directory on the server. Defaults to /.")
    parser.add_argument("host", metavar = "host", type=str,
                        help="IP or address of server to connect to.")
    parser.add_argument("--user", "-u", metavar = "user", type=str, default="anonymous",
                        help="Username. Default is 'anonymous'")
    parser.add_argument("--password", "-p", metavar = "password", type=str, default="anonymous",
                        help="Password. Default is 'anonymous'")
    return vars(parser.parse_args())


class PathExplorer():
    """
    Class responsibile for exploring the filenames and generating a Queue of files
    """
    def __init__(self, path):
        os.chdir(os.path.expanduser(path))         #This is kind of a hack, but I don't care
        self.files = []
        self._populatePaths()        
        
    def _populatePaths(self):
        """
        Create the paths and create folders as groups. The idea is that
        folders won't have to be changed as much - I don't know if this helps, but it
        certainly doesn't hurt from testing.
        """
        for (dirpath, dirnames, filenames) in os.walk("."):
            fGroup = []
            for fname in filenames:
                path = os.path.join(dirpath,fname).replace("\\","/")
                fGroup.append(path)
            self.files.append(fGroup)

    def getFileQueue(self):
        """
        Get a multiprocess Queue of filegroups to process.
        """
        st = self.files
        q = Queue()
        for cFile in st:
            q.put(cFile)
        return q


class Uploader(Process):
    """
    Responsible for actually uploading the files
    """
    def __init__(self, ftpInfo, fileQueue, baseDir, num):
        Process.__init__(self)
        self.fileQueue = fileQueue
        self.baseDir = baseDir
        self.ftpInfo = ftpInfo
        self.num = num

    def run(self):
        """
        Process the file queue sequentially
        """
        self.ftp = FTP()
        self.ftp.connect(self.ftpInfo["server"],self.ftpInfo["port"])
        self.ftp.login(self.ftpInfo["user"], self.ftpInfo["password"])
        
        while True:
            try:
                fGroup = self.fileQueue.get_nowait()
                for cFile in fGroup:
                    self.uploadFile(cFile)
            except Empty:
                break
        try:
            self.ftp.quit()
        except:
            self.ftp.close()

    def uploadFile(self, filepath):
        """
        Entry point to upload a single file
        """
        output("(%d folders remaining) [Proc %d] Going to upload %s \t" % (self.fileQueue.qsize(), self.num, filepath))
        curPath = self.ftp.pwd()
        relPath = os.path.relpath(self.baseDir + filepath, curPath).replace("\\","/")
        self.navToDirAndUpload(relPath, filepath)

    def dirList(self):
        """
        Get a list of directories that are currently valid - some sort of multicast Queue
        caching could help performance here.
        """
        validDirs = []
        def addValidDir(x):
            if x[0] == "d":
                validDirs.append(" ".join(x.split()[LSVALIDPOS:]))
        self.ftp.retrlines("LIST", callback = addValidDir)
        return validDirs

    def navToDirAndUpload(self,relPath, diskpath):
        """
        Recursively find the next folder, and if at the right level, upload.
        """
        splitPath = relPath.split("/")
        if len(splitPath) == 1:
            #Write real file
            self.ftp.storbinary("STOR %s" % splitPath[0], open(diskpath, 'rb'))
        else:
            if splitPath[0] not in self.dirList():
                if splitPath[0]!= "..":
                    #Possible race condition here?
                    try:
                        self.ftp.mkd(splitPath[0])
                    except:
                        pass
            self.ftp.cwd(splitPath[0])
            self.navToDirAndUpload("/".join(splitPath[1:]), diskpath)

if __name__ == '__main__':
    main()
