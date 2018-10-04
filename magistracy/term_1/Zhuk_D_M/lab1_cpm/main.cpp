#include <iostream>
#include <boost/property_tree/exceptions.hpp>

#include "cpm.h"

int main(int argc, char *argv[]) {
    enum ExitCode {
        NOERR,
        ERR_PARSE_JSON,
        ERR_INVALID_TASKLIST,
        ERR_UNKNOWN
    };
    int exitcode = NOERR;
    TaskList tasklist;
    TaskPathList pathlist;

    try {
        tasklist.LoadFromJsonFile("tasklist.json");
        tasklist.Dump();
        tasklist.DumpReverseTaskLinks();

        if (!tasklist.IsValid()) {
            std::cerr << "Invalid task list" << std::endl;
            return ERR_INVALID_TASKLIST;
        }

        pathlist.Create(tasklist);
        pathlist.Dump();
    } catch (const boost::property_tree::ptree_error &e) {
        std::cerr << "Boost exception: " << e.what() << std::endl;
        exitcode = ERR_PARSE_JSON;
    } catch (const std::exception &e) {
        std::cerr << "Std exception: " << e.what() << std::endl;
        exitcode = ERR_UNKNOWN;
    } catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        exitcode = ERR_UNKNOWN;
    }

    return exitcode;
}