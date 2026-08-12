// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#define _UNICODE 1
#define UNICODE 1

#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include <inttypes.h>
#include <Softpub.h>
#include <tchar.h>
#include <wincrypt.h>
#include <windows.h>
#include <wintrust.h>

#define GetProc(hModule, procName, proc) \
    (((NULL == proc) && (NULL == (*((FARPROC*)&proc) = GetProcAddress(hModule, procName)))) ? FALSE : TRUE)

typedef BOOL(WINAPI* PfnCryptMsgClose)(IN HCRYPTMSG hCryptMsg);
static PfnCryptMsgClose pfnCryptMsgClose = NULL;

typedef BOOL(WINAPI* PfnCertCloseStore)(IN HCERTSTORE hCertStore, DWORD dwFlags);
static PfnCertCloseStore pfnCertCloseStore = NULL;

typedef HCERTSTORE(WINAPI* PfnCertOpenStore)(_In_ LPCSTR lpszStoreProvider, _In_ DWORD dwEncodingType,
                                             _In_opt_ HCRYPTPROV_LEGACY hCryptProv, _In_ DWORD dwFlags,
                                             _In_opt_ const void* pvPara);
static PfnCertOpenStore pfnCertOpenStore = NULL;

typedef BOOL(WINAPI* PfnCertFreeCertificateContext)(IN PCCERT_CONTEXT pCertContext);
static PfnCertFreeCertificateContext pfnCertFreeCertificateContext = NULL;

typedef PCCERT_CONTEXT(WINAPI* PfnCertFindCertificateInStore)(IN HCERTSTORE hCertStore, IN DWORD dwCertEncodingType,
                                                              IN DWORD dwFindFlags, IN DWORD dwFindType,
                                                              IN const void* pvFindPara,
                                                              IN PCCERT_CONTEXT pPrevCertContext);
static PfnCertFindCertificateInStore pfnCertFindCertificateInStore = NULL;

typedef BOOL(WINAPI* PfnCryptMsgGetParam)(IN HCRYPTMSG hCryptMsg, IN DWORD dwParamType, IN DWORD dwIndex,
                                          OUT void* pvData, IN OUT DWORD* pcbData);
static PfnCryptMsgGetParam pfnCryptMsgGetParam = NULL;

typedef HCRYPTMSG(WINAPI* PfnCryptMsgOpenToDecode)(_In_ DWORD dwMsgEncodingType, _In_ DWORD dwFlags,
                                                   _In_ DWORD dwMsgType, _In_opt_ HCRYPTPROV_LEGACY hCryptProv,
                                                   _Reserved_ PCERT_INFO pRecipientInfo,
                                                   _In_opt_ PCMSG_STREAM_INFO pStreamInfo);
static PfnCryptMsgOpenToDecode pfnCryptMsgOpenToDecode = NULL;

typedef BOOL(WINAPI* PfnCryptMsgUpdate)(_In_ HCRYPTMSG hCryptMsg, _In_reads_bytes_opt_(cbData) const BYTE* pbData,
                                        _In_ DWORD cbData, _In_ BOOL fFinal);
static PfnCryptMsgUpdate pfnCryptMsgUpdate = NULL;

typedef BOOL(WINAPI* PfnCryptQueryObject)(DWORD dwObjectType, const void* pvObject, DWORD dwExpectedContentTypeFlags,
                                          DWORD dwExpectedFormatTypeFlags, DWORD dwFlags,
                                          DWORD* pdwMsgAndCertEncodingType, DWORD* pdwContentType, DWORD* pdwFormatType,
                                          HCERTSTORE* phCertStore, HCRYPTMSG* phMsg, const void** ppvContext);
static PfnCryptQueryObject pfnCryptQueryObject = NULL;

typedef BOOL(WINAPI* PfnCryptDecodeObjectEx)(IN DWORD dwCertEncodingType, IN LPCSTR lpszStructType,
                                             IN const BYTE* pbEncoded, IN DWORD cbEncoded, IN DWORD dwFlags,
                                             IN PCRYPT_DECODE_PARA pDecodePara, OUT void* pvStructInfo,
                                             IN OUT DWORD* pcbStructInfo);
static PfnCryptDecodeObjectEx pfnCryptDecodeObjectEx = NULL;

typedef LONG(WINAPI* PfnWinVerifyTrust)(IN HWND hwnd, IN GUID* pgActionID, IN LPVOID pWVTData);
static PfnWinVerifyTrust pfnWinVerifyTrust = NULL;

// Additional function pointer types for certificate name verification
typedef DWORD(WINAPI* PfnCertGetNameStringW)(_In_ PCCERT_CONTEXT pCertContext, _In_ DWORD dwType, _In_ DWORD dwFlags,
                                             _In_opt_ void* pvTypePara,
                                             _Out_writes_opt_(cchNameString) LPWSTR pszNameString,
                                             _In_ DWORD cchNameString);
static PfnCertGetNameStringW pfnCertGetNameStringW = NULL;

// Thread-safe initialization flags
static std::once_flag g_crypt32InitFlag;
static std::once_flag g_wintrustInitFlag;
static bool g_crypt32Initialized = false;
static bool g_wintrustInitialized = false;

//! \brief Thread-safe initialization function for crypt32.dll function pointers
//! \return true if initialization succeeded, false otherwise
inline bool initializeCrypt32Functions()
{
    std::call_once(g_crypt32InitFlag,
                   []()
                   {
                       // We only support Win10+ so we can search for module in system32 directly
                       auto hModCrypt32 = LoadLibraryExW(L"crypt32.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
                       if (hModCrypt32 && GetProc(hModCrypt32, "CryptMsgClose", pfnCryptMsgClose) &&
                           GetProc(hModCrypt32, "CertCloseStore", pfnCertCloseStore) &&
                           GetProc(hModCrypt32, "CertOpenStore", pfnCertOpenStore) &&
                           GetProc(hModCrypt32, "CertFreeCertificateContext", pfnCertFreeCertificateContext) &&
                           GetProc(hModCrypt32, "CertFindCertificateInStore", pfnCertFindCertificateInStore) &&
                           GetProc(hModCrypt32, "CryptMsgGetParam", pfnCryptMsgGetParam) &&
                           GetProc(hModCrypt32, "CryptMsgUpdate", pfnCryptMsgUpdate) &&
                           GetProc(hModCrypt32, "CryptMsgOpenToDecode", pfnCryptMsgOpenToDecode) &&
                           GetProc(hModCrypt32, "CryptQueryObject", pfnCryptQueryObject) &&
                           GetProc(hModCrypt32, "CryptDecodeObjectEx", pfnCryptDecodeObjectEx) &&
                           GetProc(hModCrypt32, "CertGetNameStringW", pfnCertGetNameStringW))
                       {
                           g_crypt32Initialized = true;
                           // NOTE: FreeLibrary is intentionally not called on hModCrypt32.
                           // The library is loaded once and kept for the lifetime of the process because:
                           // 1. The function pointers (pfnCrypt*, pfnCert*) must remain valid for all subsequent calls
                           // 2. System DLLs like crypt32.dll are reference-counted and shared across processes
                           // 3. Avoiding reload overhead improves performance for repeated cryptographic operations
                           // 4. The OS automatically unloads all DLLs when the process terminates
                       }
                   });
    return g_crypt32Initialized;
}

//! \brief Thread-safe initialization function for wintrust.dll function pointers
//! \return true if initialization succeeded, false otherwise
inline bool initializeWintrustFunctions()
{
    std::call_once(g_wintrustInitFlag,
                   []()
                   {
                       // We only support Win10+ so we can search for module in system32 directly
                       auto hModWintrust = LoadLibraryExW(L"wintrust.dll", NULL, LOAD_LIBRARY_SEARCH_SYSTEM32);
                       if (hModWintrust && GetProc(hModWintrust, "WinVerifyTrust", pfnWinVerifyTrust))
                       {
                           g_wintrustInitialized = true;
                           // NOTE: FreeLibrary is intentionally not called on hModWintrust.
                           // The library is loaded once and kept for the lifetime of the process because:
                           // 1. The function pointers (pfnWinVerifyTrust) must remain valid for all subsequent calls
                           // 2. System DLLs like wintrust.dll are reference-counted and shared across processes
                           // 3. Avoiding reload overhead improves performance for repeated signature verification
                           // 4. The OS automatically unloads all DLLs when the process terminates
                       }
                   });
    return g_wintrustInitialized;
}

namespace trt_rtx_ep
{

namespace security
{

// Constants for certificate name buffer size
constexpr DWORD kMaxSignerInfoSize = 1024 * 1024;  // 1 MB - maximum safe size for signer info
constexpr DWORD kMaxSubjectNameLength = 256;       // Maximum length for certificate subject name

//! \brief Verify that the primary signer of a file is NVIDIA Corporation
//! \param pathToFile Full path to the file to verify
//! \return true if the file is signed by NVIDIA Corporation, false otherwise
//! \note This function checks the certificate subject name to verify the signer identity
bool verifyPrimarySignerIsNVIDIA(const wchar_t* pathToFile)
{
    bool valid = false;

    // Verify the primary signature is from NVIDIA Corporation by checking the certificate subject name

    // Initialize crypt32.dll function pointers in a thread-safe manner
    if (!initializeCrypt32Functions())
    {
        return false;
    }

    DWORD dwEncoding, dwContentType, dwFormatType;
    HCERTSTORE hStore = NULL;
    HCRYPTMSG hMsg = NULL;
    PCCERT_CONTEXT pCertContext = NULL;

    // Get message handle, store handle, and signer certificate from the signed file.
    auto bResult = pfnCryptQueryObject(CERT_QUERY_OBJECT_FILE, pathToFile, CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED,
                                       CERT_QUERY_FORMAT_FLAG_BINARY, 0, &dwEncoding, &dwContentType, &dwFormatType,
                                       &hStore, &hMsg, NULL);
    if (!bResult)
    {
        return false;
    }

    // Get the signer certificate from the message
    DWORD dwSignerInfo = 0;
    bResult = pfnCryptMsgGetParam(hMsg, CMSG_SIGNER_INFO_PARAM, 0, NULL, &dwSignerInfo);

    if (bResult && dwSignerInfo > 0)
    {
        // Security check: Validate dwSignerInfo to prevent integer overflow attacks
        // CMSG_SIGNER_INFO is typically a few KB; 1MB is a generous safe maximum
        if (dwSignerInfo > kMaxSignerInfoSize)
        {
            pfnCryptMsgClose(hMsg);
            pfnCertCloseStore(hStore, CERT_CLOSE_STORE_FORCE_FLAG);
            return false;
        }

        // Use RAII wrapper for LocalAlloc/LocalFree
        struct LocalAllocDeleter
        {
            void operator()(void* ptr) const
            {
                if (ptr)
                {
                    LocalFree(ptr);
                }
            }
        };
        using SignerInfoPtr = std::unique_ptr<CMSG_SIGNER_INFO, LocalAllocDeleter>;

        SignerInfoPtr pSignerInfo(static_cast<PCMSG_SIGNER_INFO>(LocalAlloc(LPTR, dwSignerInfo)));
        if (pSignerInfo)
        {
            if (pfnCryptMsgGetParam(hMsg, CMSG_SIGNER_INFO_PARAM, 0, static_cast<PVOID>(pSignerInfo.get()),
                                    &dwSignerInfo))
            {
                // Find the signer certificate in the store
                CERT_INFO CertInfo{};
                CertInfo.Issuer = pSignerInfo->Issuer;
                CertInfo.SerialNumber = pSignerInfo->SerialNumber;

                pCertContext =
                    pfnCertFindCertificateInStore(hStore, (X509_ASN_ENCODING | PKCS_7_ASN_ENCODING), 0,
                                                  CERT_FIND_SUBJECT_CERT, static_cast<PVOID>(&CertInfo), NULL);

                if (pCertContext)
                {
                    // Get the certificate subject name
                    wchar_t subjectName[kMaxSubjectNameLength];
                    DWORD nameLen = pfnCertGetNameStringW(pCertContext, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, NULL,
                                                          subjectName, kMaxSubjectNameLength);

                    if (nameLen > 1)
                    {
                        // Check if subject contains "NVIDIA Corporation"
                        valid = (wcsstr(subjectName, L"NVIDIA Corporation") != NULL);
                    }
                }
            }
            // pSignerInfo automatically freed by unique_ptr destructor
        }
    }

    // Cleanup
    if (pCertContext)
    {
        pfnCertFreeCertificateContext(pCertContext);
    }
    if (hMsg)
    {
        pfnCryptMsgClose(hMsg);
    }
    if (hStore)
    {
        pfnCertCloseStore(hStore, CERT_CLOSE_STORE_FORCE_FLAG);
    }

    return valid;
}

//! \brief Verify that a file has a valid embedded signature
//! \param pathToFile Full path to the file to verify (must be absolute path)
//! \return true if the file has a valid embedded signature, false otherwise
//! \note See
//! https://docs.microsoft.com/en-us/windows/win32/seccrypto/example-c-program--verifying-the-signature-of-a-pe-file
//! \note IMPORTANT: Always pass in the FULL PATH to the file, relative paths are NOT allowed!
//!
//! WVTPolicyGUID specifies the policy to apply on the file.
//! WINTRUST_ACTION_GENERIC_VERIFY_V2 policy checks:
//! 1) The certificate used to sign the file chains up to a root certificate located in the trusted root certificate
//! store.
//!    This implies that the identity of the publisher has been verified by a certification authority.
//! 2) In cases where user interface is displayed (which this example does not do), WinVerifyTrust will check for
//! whether
//!    the end entity certificate is stored in the trusted publisher store, implying that the user trusts content from
//!    this publisher.
//! 3) The end entity certificate has sufficient permission to sign code, as indicated by the presence of a code signing
//! EKU or no EKU.
bool verifyEmbeddedSignature(const wchar_t* pathToFile)
{
    bool valid = true;

    LONG lStatus = {};

    // Initialize the WINTRUST_FILE_INFO structure.
    WINTRUST_FILE_INFO FileData{};
    FileData.cbStruct = sizeof(WINTRUST_FILE_INFO);
    FileData.pcwszFilePath = pathToFile;
    FileData.hFile = NULL;
    FileData.pgKnownSubject = NULL;

    // Initialize wintrust.dll function pointers in a thread-safe manner
    if (!initializeWintrustFunctions())
    {
        return false;
    }

    GUID WVTPolicyGUID = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    WINTRUST_DATA WinTrustData{};

    // Initialize the WinVerifyTrust input data structure.
    WinTrustData.cbStruct = sizeof(WinTrustData);
    // Use default code signing EKU.
    WinTrustData.pPolicyCallbackData = NULL;
    // No data to pass to SIP.
    WinTrustData.pSIPClientData = NULL;
    // Disable WVT UI.
    WinTrustData.dwUIChoice = WTD_UI_NONE;
    // No revocation checking.
    WinTrustData.fdwRevocationChecks = WTD_REVOKE_NONE;
    // Verify an embedded signature on a file.
    WinTrustData.dwUnionChoice = WTD_CHOICE_FILE;
    // Verify action.
    WinTrustData.dwStateAction = WTD_STATEACTION_VERIFY;
    // Verification sets this value.
    WinTrustData.hWVTStateData = NULL;
    // Not used.
    WinTrustData.pwszURLReference = NULL;
    // This is not applicable if there is no UI because it changes
    // the UI to accommodate running applications instead of
    // installing applications.
    WinTrustData.dwUIContext = 0;
    // Set pFile.
    WinTrustData.pFile = &FileData;

    // Verify the primary signature with strong cryptography enforcement
    CERT_STRONG_SIGN_PARA StrongSigPolicy{};
    StrongSigPolicy.cbSize = sizeof(CERT_STRONG_SIGN_PARA);
    StrongSigPolicy.dwInfoChoice = CERT_STRONG_SIGN_OID_INFO_CHOICE;
    StrongSigPolicy.pszOID = const_cast<LPSTR>(szOID_CERT_STRONG_SIGN_OS_CURRENT);

    WINTRUST_SIGNATURE_SETTINGS SignatureSettings{};
    SignatureSettings.cbStruct = sizeof(WINTRUST_SIGNATURE_SETTINGS);
    SignatureSettings.pCryptoPolicy = &StrongSigPolicy;
    WinTrustData.pSignatureSettings = &SignatureSettings;

    // WinVerifyTrust verifies signatures as specified by the GUID and Wintrust_Data.
    lStatus = pfnWinVerifyTrust(NULL, &WVTPolicyGUID, &WinTrustData);

    // Signature must be validated by the OS
    valid = lStatus == ERROR_SUCCESS;
    if (valid)
    {
        // Verify the primary signer is NVIDIA Corporation
        valid &= verifyPrimarySignerIsNVIDIA(pathToFile);
    }

    // Any hWVTStateData must be released by a call with close.
    WinTrustData.dwStateAction = WTD_STATEACTION_CLOSE;
    lStatus = pfnWinVerifyTrust(NULL, &WVTPolicyGUID, &WinTrustData);

    return valid;
}

}  // namespace security
}  // namespace trt_rtx_ep

// =============================================================================
// Public API for signature verification
// =============================================================================

/// @brief Verify that a file is signed by NVIDIA Corporation
/// @param pathToFile Full path to the file to verify (must be absolute path)
/// @return true if the file has a valid NVIDIA Corporation signature, false otherwise
/// @note This function verifies:
///       1. The signature chains to a trusted CA (OS verification with strong crypto)
///       2. The signer certificate subject contains "NVIDIA Corporation"
///       This approach survives certificate/key rotation and is ISV-friendly.
inline bool VerifyNvidiaSignature(const std::wstring& pathToFile)
{
    return trt_rtx_ep::security::verifyEmbeddedSignature(pathToFile.c_str());
}
